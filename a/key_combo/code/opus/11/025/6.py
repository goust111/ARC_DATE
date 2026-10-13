from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_opus_11_025_6",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


BACKGROUND = Color.BLACK
OBJECT = Color.RED
STAMP = Color.GREEN
MARKER = Color.PINK

DIAG_NW_SE = 0
DIAG_NE_SW = 1

THRESHOLD_CHOICES = (8, 9)
GRID_HEIGHT_RANGE = (14, 20)
GRID_WIDTH_RANGE = (16, 22)
OBJECT_COUNT_RANGE = (4, 6)
BLOCK_SIZE_CHOICES = (1, 2, 3)
GENERATION_RETRIES = 400


def _square_patch(size: int, color: int) -> np.ndarray:
    return np.full((int(size), int(size)), int(color), dtype=int)


def _rect_cells(r0: int, c0: int, h: int, w: int, limit_h: int, limit_w: int) -> set[tuple[int, int]]:
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(int(limit_h), int(r0) + int(h))
    cc1 = min(int(limit_w), int(c0) + int(w))
    if rr0 >= rr1 or cc0 >= cc1:
        return set()
    return {(r, c) for r in range(rr0, rr1) for c in range(cc0, cc1)}


def _decode_threshold(grid: np.ndarray) -> int:
    bottom = np.asarray(grid[-1], dtype=int)
    marker_cols = np.flatnonzero(bottom == MARKER)
    assert marker_cols.size > 0, "expected a non-empty marker run on the bottom row"
    assert np.all(bottom[bottom != MARKER] == BACKGROUND), "marker row must be black except for the marker run"
    assert int(marker_cols[-1] - marker_cols[0] + 1) == int(marker_cols.size), "marker pixels must form one contiguous run"
    return int(marker_cols.size)


def _target_origins(r0: int, c0: int, size: int, diagonal: int) -> list[tuple[int, int]]:
    if int(diagonal) == DIAG_NW_SE:
        return [(r0 - size, c0 + 2 * size), (r0 + 2 * size, c0 - size)]
    return [(r0 - size, c0 - size), (r0 + 2 * size, c0 + 2 * size)]


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all(np.isin(g[:-1], [BACKGROUND, OBJECT])), "main area must contain only black or red"
    assert np.all(np.isin(g[-1], [BACKGROUND, MARKER])), "bottom row must contain only black and marker pixels"

    components = find_components(
        g[:-1],
        bg=BACKGROUND,
        colors=[OBJECT],
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    assert components, "expected at least one red motif"

    objects: list[dict] = []
    for comp in components:
        coords = comp["coords"]
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h == w and h % 2 == 0, "each red motif must occupy a square 2x2 block footprint"
        size = h // 2
        sub = g[r0 : r1 + 1, c0 : c1 + 1]

        occupied: set[tuple[int, int]] = set()
        for br in range(2):
            for bc in range(2):
                block = sub[br * size : (br + 1) * size, bc * size : (bc + 1) * size]
                if np.all(block == OBJECT):
                    occupied.add((br, bc))
                else:
                    assert np.all(block == BACKGROUND), "only two diagonal blocks may be red"

        if occupied == {(0, 0), (1, 1)}:
            diagonal = DIAG_NW_SE
        else:
            assert occupied == {(0, 1), (1, 0)}, "filled blocks must be diagonal"
            diagonal = DIAG_NE_SW

        area = int(coords.shape[0])
        assert area == 2 * size * size
        target_origins = _target_origins(r0, c0, size, diagonal)
        target_cells: set[tuple[int, int]] = set()
        for tr, tc in target_origins:
            clipped = _rect_cells(tr, tc, size, size, g.shape[0] - 1, g.shape[1])
            assert clipped, "each outward stamp must be at least partially visible"
            target_cells |= clipped

        objects.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "size": size,
                "area": area,
                "diagonal": diagonal,
                "target_origins": target_origins,
                "target_cells": target_cells,
            }
        )

    return sorted(objects, key=lambda obj: obj["bbox"])


def main(grid: np.ndarray) -> np.ndarray:
    """Decode the bottom marker threshold, keep only large enough diagonal red motifs, and stamp the opposite outward diagonal in green.

    Input structure:
    - The grid has a black background and a bottom marker row.
    - Above the marker row are red diagonal motifs: each motif occupies a square 2x2 block footprint of equal square blocks, with exactly two red blocks on one diagonal and the other two blocks black.
    - The bottom row contains one contiguous magenta run whose length encodes the area threshold.

    Rule:
    - Read the number of magenta pixels in the bottom row to get threshold T.
    - Find every red diagonal motif, measure its red area, and delete motifs with area < T.
    - Keep motifs with area >= T and stamp green square blocks of the same size at the two outward corners of the complementary diagonal, clipped at the grid boundary; leave the bottom marker row unchanged.

    Key insight:
    - The transformation depends on three linked steps: the magenta run supplies a variable threshold, object area determines which motifs survive, and only the surviving motifs receive the outward complementary diagonal stamps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3
    threshold = _decode_threshold(g)
    objects = _extract_objects(g)

    out = np.full_like(g, BACKGROUND)
    out[-1, :] = g[-1, :]

    for obj in objects:
        if int(obj["area"]) < int(threshold):
            continue

        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = OBJECT
        for tr, tc in obj["target_origins"]:
            paste(out, _square_patch(obj["size"], STAMP), int(tr), int(tc), clip=True)

    out[-1, :] = g[-1, :]
    return out


def _make_candidate(height: int, width: int, r0: int, c0: int, size: int, diagonal: int) -> dict | None:
    draw_h = int(height) - 1
    if r0 < 0 or c0 < 0 or r0 + 2 * size > draw_h or c0 + 2 * size > int(width):
        return None

    if int(diagonal) == DIAG_NW_SE:
        red_origins = [(r0, c0), (r0 + size, c0 + size)]
    else:
        red_origins = [(r0, c0 + size), (r0 + size, c0)]

    bbox = (r0, c0, r0 + 2 * size - 1, c0 + 2 * size - 1)
    bbox_cells = _rect_cells(r0, c0, 2 * size, 2 * size, draw_h, width)
    bbox_margin_cells = _rect_cells(r0 - 1, c0 - 1, 2 * size + 2, 2 * size + 2, draw_h, width)
    target_origins = _target_origins(r0, c0, size, diagonal)

    clipped = False
    target_cells: set[tuple[int, int]] = set()
    for tr, tc in target_origins:
        vis = _rect_cells(tr, tc, size, size, draw_h, width)
        if not vis:
            return None
        target_cells |= vis
        if tr < 0 or tc < 0 or tr + size > draw_h or tc + size > int(width):
            clipped = True

    return {
        "bbox": bbox,
        "bbox_cells": bbox_cells,
        "bbox_margin_cells": bbox_margin_cells,
        "size": int(size),
        "diagonal": int(diagonal),
        "area": int(2 * size * size),
        "red_origins": red_origins,
        "target_origins": target_origins,
        "target_cells": target_cells,
        "clipped": clipped,
    }


def _candidate_positions(height: int, width: int, size: int, diagonal: int, require_clip: bool | None) -> list[dict]:
    options: list[dict] = []
    max_r0 = int(height) - 1 - 2 * int(size)
    max_c0 = int(width) - 2 * int(size)
    for r0 in range(max_r0 + 1):
        for c0 in range(max_c0 + 1):
            candidate = _make_candidate(height, width, r0, c0, size, diagonal)
            if candidate is None:
                continue
            if require_clip is True and not bool(candidate["clipped"]):
                continue
            if require_clip is False and bool(candidate["clipped"]):
                continue
            options.append(candidate)
    random.shuffle(options)
    return options


def _fits(candidate: dict, state: dict) -> bool:
    if candidate["bbox_cells"] & state["bbox_margin_cells"]:
        return False
    if candidate["bbox_margin_cells"] & state["bbox_cells"]:
        return False
    if candidate["bbox_cells"] & state["target_cells"]:
        return False
    if candidate["target_cells"] & state["bbox_margin_cells"]:
        return False
    if candidate["target_cells"] & state["target_cells"]:
        return False
    return True


def _place_objects(height: int, width: int, specs: list[dict], clip_requirements: list[bool | None]) -> list[dict] | None:
    candidate_lists: list[list[dict]] = []
    for i, spec in enumerate(specs):
        candidates = _candidate_positions(height, width, spec["size"], spec["diagonal"], clip_requirements[i])
        if not candidates:
            return None
        candidate_lists.append(candidates)

    order = sorted(range(len(specs)), key=lambda i: (len(candidate_lists[i]), -int(specs[i]["size"])))
    chosen: list[dict | None] = [None] * len(specs)

    def rec(k: int, state: dict) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        for candidate in candidate_lists[idx]:
            if not _fits(candidate, state):
                continue
            chosen[idx] = candidate
            next_state = {
                "bbox_cells": state["bbox_cells"] | candidate["bbox_cells"],
                "bbox_margin_cells": state["bbox_margin_cells"] | candidate["bbox_margin_cells"],
                "target_cells": state["target_cells"] | candidate["target_cells"],
            }
            if rec(k + 1, next_state):
                return True
            chosen[idx] = None
        return False

    initial_state = {"bbox_cells": set(), "bbox_margin_cells": set(), "target_cells": set()}
    if not rec(0, initial_state):
        return None
    return [obj for obj in chosen if obj is not None]


def _build_specs(threshold: int, count: int) -> list[dict]:
    sizes = [1, 2, 3]
    while len(sizes) < int(count):
        sizes.append(random.choice([1, 1, 2, 2, 3]))
    random.shuffle(sizes)

    diagonals = [DIAG_NW_SE, DIAG_NE_SW]
    while len(diagonals) < int(count):
        diagonals.append(random.choice([DIAG_NW_SE, DIAG_NE_SW]))
    random.shuffle(diagonals)

    specs = []
    for i in range(int(count)):
        size = int(sizes[i])
        specs.append(
            {
                "size": size,
                "diagonal": int(diagonals[i]),
                "area": int(2 * size * size),
                "retained": bool(2 * size * size >= int(threshold)),
            }
        )
    return specs


def _render_input(height: int, width: int, threshold: int, placed: list[dict]) -> np.ndarray:
    grid = np.full((int(height), int(width)), BACKGROUND, dtype=int)
    for obj in placed:
        for r0, c0 in obj["red_origins"]:
            paste(grid, _square_patch(obj["size"], OBJECT), int(r0), int(c0))

    start = random.randint(0, int(width) - int(threshold))
    grid[-1, start : start + int(threshold)] = MARKER
    return grid


def _validate_generated_input(grid: np.ndarray) -> None:
    threshold = _decode_threshold(grid)
    objects = _extract_objects(grid)
    assert OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]
    assert {int(obj["diagonal"]) for obj in objects} == {DIAG_NW_SE, DIAG_NE_SW}
    assert any(int(obj["area"]) < int(threshold) for obj in objects), "expected at least one deleted object"
    assert any(int(obj["area"]) >= int(threshold) for obj in objects), "expected at least one retained object"
    assert any(int(obj["area"]) == 8 for obj in objects), "expected at least one threshold-sensitive size-2 motif"
    assert any(int(obj["area"]) == 18 for obj in objects), "expected at least one size-3 motif"
    if int(threshold) == 9:
        assert any(int(obj["area"]) == 8 for obj in objects) and any(int(obj["area"]) == 18 for obj in objects)

    out = main(grid)
    assert np.array_equal(out[-1], grid[-1]), "marker row must remain unchanged"

    expected_green = set()
    for obj in objects:
        coords = obj["coords"]
        if int(obj["area"]) < int(threshold):
            assert np.all(out[coords[:, 0], coords[:, 1]] == BACKGROUND), "small objects must be erased"
        else:
            assert np.all(out[coords[:, 0], coords[:, 1]] == OBJECT), "large objects must remain red"
            expected_green |= set(obj["target_cells"])

    actual_green = {tuple(map(int, cell)) for cell in np.argwhere(out == STAMP)}
    assert actual_green == expected_green, "green stamps must match retained motifs exactly"


def _gen_attempt() -> np.ndarray | None:
    threshold = random.choice(THRESHOLD_CHOICES)
    height = random.randint(max(GRID_HEIGHT_RANGE[0], 2 * max(BLOCK_SIZE_CHOICES) + 2), GRID_HEIGHT_RANGE[1])
    width = random.randint(max(GRID_WIDTH_RANGE[0], threshold + 2), GRID_WIDTH_RANGE[1])
    count = random.randint(*OBJECT_COUNT_RANGE)

    specs = _build_specs(threshold, count)
    retained_indices = [i for i, spec in enumerate(specs) if spec["retained"]]
    if not retained_indices:
        return None

    clip_requirements: list[bool | None] = [None] * len(specs)
    if random.choice([False, True]):
        clip_index = random.choice(retained_indices)
        clip_requirements[clip_index] = True
    else:
        for i in retained_indices:
            clip_requirements[i] = False

    placed = _place_objects(height, width, specs, clip_requirements)
    if placed is None:
        return None

    grid = _render_input(height, width, threshold, placed)
    try:
        _validate_generated_input(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
