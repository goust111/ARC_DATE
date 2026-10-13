from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MARKER = 5
BLOB_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_HEIGHT_RANGE = (17, 25)
GRID_WIDTH_RANGE = (17, 25)
BLOB_COUNT_RANGE = (3, 5)
SPAN_CHOICES = (1, 3, 3, 5, 5, 7)
RESERVE_PAD = 1
GENERATION_RETRIES = 800


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _sample_cells(bbox: tuple[int, int, int, int], parity: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pr, pc = parity
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r % 2, c % 2) == (pr, pc)
    }


def _parse_scene(grid: np.ndarray) -> tuple[tuple[int, int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    marker_coords = np.argwhere(g == MARKER)
    assert marker_coords.shape == (1, 2), "expected exactly one grey marker pixel"
    marker = (int(marker_coords[0, 0]), int(marker_coords[0, 1]))

    blobs: list[dict[str, object]] = []
    for color in sorted(int(x) for x in np.unique(g) if x not in (BACKGROUND, MARKER)):
        coords = np.argwhere(g == color)
        assert coords.size > 0, "each blob color must appear"
        row_parities = {int(v % 2) for v in coords[:, 0]}
        col_parities = {int(v % 2) for v in coords[:, 1]}
        assert len(row_parities) == 1 and len(col_parities) == 1, "blob must occupy one parity lattice"
        bbox = _bbox_coords(coords)
        blobs.append(
            {
                "color": color,
                "coords": coords,
                "bbox": bbox,
                "parity": (row_parities.pop(), col_parities.pop()),
            }
        )

    assert blobs, "expected at least one colored remnant blob"
    return marker, blobs


def main(grid: np.ndarray) -> np.ndarray:
    """Use the grey pixel's coordinate parity to decide which dotted remnants become solid rectangles.

    Input structure
    - A black grid contains several colored dotted remnants plus one isolated grey marker pixel.
    - Each colored remnant consists only of cells on one fixed `(row parity, col parity)` lattice, as if a solid rectangle had been sampled every 2 cells.
    - Different remnants may use different parity lattices, but the grey marker is not part of any remnant.

    Rule
    - Read the grey marker's `(row % 2, col % 2)` as the selected parity key.
    - For every colored remnant whose dots all lie on that same parity lattice, fill the bounding box of its visible dots with the remnant color.
    - Erase every remnant on the other three parity lattices back to black, and keep the grey marker pixel.

    Key insight
    - The grey pixel is not a spatial anchor: it encodes which parity lattice survives, so both row parity and column parity are jointly necessary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker, blobs = _parse_scene(g)
    selected = (marker[0] % 2, marker[1] % 2)

    out = np.full(g.shape, BACKGROUND, dtype=int)
    for blob in blobs:
        if blob["parity"] != selected:
            continue
        r0, c0, r1, c1 = blob["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(blob["color"])

    out[marker] = MARKER
    return out


def _sample_spans() -> tuple[int, int]:
    while True:
        h = random.choice(SPAN_CHOICES)
        w = random.choice(SPAN_CHOICES)
        if h != 1 or w != 1:
            return h, w


def _sample_blob_specs(selected: tuple[int, int]) -> list[dict[str, object]]:
    pr, pc = selected
    row_only = (pr, 1 - pc)
    col_only = (1 - pr, pc)
    opposite = (1 - pr, 1 - pc)

    blob_count = random.randint(*BLOB_COUNT_RANGE)
    parities = [selected, row_only, col_only]
    extra_pool = [selected, selected, opposite, opposite, row_only, col_only]
    while len(parities) < blob_count:
        parities.append(random.choice(extra_pool))
    random.shuffle(parities)

    colors = random.sample(BLOB_COLORS, blob_count)
    specs: list[dict[str, object]] = []
    for color, parity in zip(colors, parities):
        h, w = _sample_spans()
        specs.append({"color": color, "parity": parity, "h": h, "w": w})
    return specs


def _place_blobs(height: int, width: int, specs: list[dict[str, object]]) -> bool:
    reserved = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: int(specs[i]["h"]) * int(specs[i]["w"]), reverse=True)

    for idx in order:
        spec = specs[idx]
        h = int(spec["h"])
        w = int(spec["w"])
        pr, pc = spec["parity"]
        candidates: list[tuple[int, int]] = []
        for r0 in range(0, height - h + 1):
            if r0 % 2 != pr:
                continue
            for c0 in range(0, width - w + 1):
                if c0 % 2 != pc:
                    continue
                rr0 = max(0, r0 - RESERVE_PAD)
                cc0 = max(0, c0 - RESERVE_PAD)
                rr1 = min(height, r0 + h + RESERVE_PAD)
                cc1 = min(width, c0 + w + RESERVE_PAD)
                if np.any(reserved[rr0:rr1, cc0:cc1]):
                    continue
                candidates.append((r0, c0))

        if not candidates:
            return False

        r0, c0 = random.choice(candidates)
        spec["bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)
        rr0 = max(0, r0 - RESERVE_PAD)
        cc0 = max(0, c0 - RESERVE_PAD)
        rr1 = min(height, r0 + h + RESERVE_PAD)
        cc1 = min(width, c0 + w + RESERVE_PAD)
        reserved[rr0:rr1, cc0:cc1] = True

    return True


def _sample_marker(height: int, width: int, selected: tuple[int, int], specs: list[dict[str, object]]) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    for r in range(1, height - 1):
        if r % 2 != selected[0]:
            continue
        for c in range(1, width - 1):
            if c % 2 != selected[1]:
                continue
            clear = True
            for spec in specs:
                r0, c0, r1, c1 = spec["bbox"]
                if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                    clear = False
                    break
            if clear:
                candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _render_input(height: int, width: int, specs: list[dict[str, object]], marker: tuple[int, int]) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for spec in specs:
        cells = _sample_cells(spec["bbox"], spec["parity"])
        for r, c in cells:
            grid[r, c] = int(spec["color"])
    grid[marker] = MARKER
    return grid


def _expected_output(shape: tuple[int, int], specs: list[dict[str, object]], marker: tuple[int, int]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    selected = (marker[0] % 2, marker[1] % 2)
    for spec in specs:
        if spec["parity"] != selected:
            continue
        r0, c0, r1, c1 = spec["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
    out[marker] = MARKER
    return out


def _shortcut_output(grid: np.ndarray, mode: str) -> np.ndarray:
    marker, blobs = _parse_scene(grid)
    mr, mc = marker[0] % 2, marker[1] % 2
    out = np.full(grid.shape, BACKGROUND, dtype=int)

    for blob in blobs:
        br, bc = blob["parity"]
        keep = False
        if mode == "all":
            keep = True
        elif mode == "row":
            keep = br == mr
        elif mode == "col":
            keep = bc == mc
        else:
            raise ValueError(f"unknown mode: {mode}")

        if keep:
            r0, c0, r1, c1 = blob["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = int(blob["color"])

    out[marker] = MARKER
    return out


def _validate_case(inp: np.ndarray, specs: list[dict[str, object]], marker: tuple[int, int]) -> bool:
    expected = _expected_output(inp.shape, specs, marker)
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(inp, expected):
        return False

    parsed_marker, blobs = _parse_scene(inp)
    if parsed_marker != marker or len(blobs) != len(specs):
        return False

    selected = (marker[0] % 2, marker[1] % 2)
    row_only = (selected[0], 1 - selected[1])
    col_only = (1 - selected[0], selected[1])
    by_color = {int(spec["color"]): spec for spec in specs}

    selected_count = 0
    erased_count = 0
    has_row_only = False
    has_col_only = False
    completion_gain = 0

    for blob in blobs:
        color = int(blob["color"])
        spec = by_color.get(color)
        if spec is None:
            return False
        if blob["bbox"] != spec["bbox"] or blob["parity"] != spec["parity"]:
            return False

        coords = {tuple(map(int, coord)) for coord in blob["coords"]}
        if coords != _sample_cells(spec["bbox"], spec["parity"]):
            return False

        r0, c0, r1, c1 = blob["bbox"]
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if blob["parity"] == selected:
            selected_count += 1
            completion_gain += bbox_area - len(coords)
        else:
            erased_count += 1
        if blob["parity"] == row_only:
            has_row_only = True
        if blob["parity"] == col_only:
            has_col_only = True

    if selected_count < 1 or erased_count < 2:
        return False
    if not has_row_only or not has_col_only:
        return False
    if completion_gain <= 0:
        return False

    if np.array_equal(expected, _shortcut_output(inp, "all")):
        return False
    if np.array_equal(expected, _shortcut_output(inp, "row")):
        return False
    if np.array_equal(expected, _shortcut_output(inp, "col")):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    selected = (random.randint(0, 1), random.randint(0, 1))
    specs = _sample_blob_specs(selected)

    if not _place_blobs(height, width, specs):
        return None

    marker = _sample_marker(height, width, selected, specs)
    if marker is None:
        return None

    inp = _render_input(height, width, specs, marker)
    if not _validate_case(inp, specs, marker):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid parity-remnant reconstruction input")
