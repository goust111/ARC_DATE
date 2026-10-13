from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
ORIENTATIONS = ("h", "v", "d1", "d2")
STEPS = {
    "h": (0, 1),
    "v": (1, 0),
    "d1": (1, 1),
    "d2": (1, -1),
}

# Generator configuration.
GRID_H_RANGE = (10, 19)
GRID_W_RANGE = (10, 19)
LINE_COUNT_RANGE = (3, 6)
MIN_LINE_LENGTH = 3
MAX_GENERATION_ATTEMPTS = 400
LINE_PLACEMENT_ATTEMPTS = 200
SCENE_ATTEMPTS = 120
MAX_CROSSINGS_PER_SCENE = 6


def _as_color_set(grid: np.ndarray) -> list[int]:
    return sorted(int(color) for color in np.unique(grid) if int(color) != BG)


def _infer_orientation(coords: np.ndarray) -> str:
    rows = coords[:, 0]
    cols = coords[:, 1]
    candidates: list[str] = []
    if np.all(rows == rows[0]):
        candidates.append("h")
    if np.all(cols == cols[0]):
        candidates.append("v")
    if np.all(rows - cols == rows[0] - cols[0]):
        candidates.append("d1")
    if np.all(rows + cols == rows[0] + cols[0]):
        candidates.append("d2")
    assert len(candidates) == 1, f"ambiguous line orientation for coords={coords.tolist()}"
    return candidates[0]


def _segment_from_visible(coords: np.ndarray, orientation: str) -> np.ndarray:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if orientation == "h":
        row = int(rows[0])
        c0, c1 = int(cols.min()), int(cols.max())
        return np.array([(row, c) for c in range(c0, c1 + 1)], dtype=int)
    if orientation == "v":
        col = int(cols[0])
        r0, r1 = int(rows.min()), int(rows.max())
        return np.array([(r, col) for r in range(r0, r1 + 1)], dtype=int)
    if orientation == "d1":
        delta = int(rows[0] - cols[0])
        r0, r1 = int(rows.min()), int(rows.max())
        return np.array([(r, r - delta) for r in range(r0, r1 + 1)], dtype=int)

    total = int(rows[0] + cols[0])
    r0, r1 = int(rows.min()), int(rows.max())
    return np.array([(r, total - r) for r in range(r0, r1 + 1)], dtype=int)


def _cell_key(cell: Iterable[int]) -> tuple[int, int]:
    r, c = cell
    return int(r), int(c)


def _extract_lines(grid: np.ndarray) -> list[dict[str, object]]:
    lines: list[dict[str, object]] = []
    for color in _as_color_set(grid):
        coords = np.argwhere(grid == color)
        assert len(coords) >= 2, f"line color {color} must expose at least two pixels"
        orientation = _infer_orientation(coords)
        full_coords = _segment_from_visible(coords, orientation)
        lines.append(
            {
                "color": int(color),
                "orientation": orientation,
                "coords": full_coords,
                "length": int(len(full_coords)),
            }
        )
    lengths = [int(line["length"]) for line in lines]
    assert len(lengths) == len(set(lengths)), "line lengths must be unique"
    return lines


def _coverage(lines: list[dict[str, object]]) -> dict[tuple[int, int], list[int]]:
    cover: dict[tuple[int, int], list[int]] = {}
    for idx, line in enumerate(lines):
        for cell in np.asarray(line["coords"], dtype=int):
            cover.setdefault(_cell_key(cell), []).append(idx)
    return cover


def main(grid: np.ndarray) -> np.ndarray:
    """Repaint overlapping straight lines so longer lines cover shorter ones at every crossing."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    lines = _extract_lines(g)
    cover = _coverage(lines)

    for r, c in np.argwhere(g != BG):
        cell = (int(r), int(c))
        assert cell in cover, f"visible pixel {cell} is not explained by any recovered line"
        visible = int(g[r, c])
        assert any(int(lines[idx]["color"]) == visible for idx in cover[cell]), (
            f"visible color {visible} at {cell} is not carried by any recovered line"
        )

    for idx, line in enumerate(lines):
        for r, c in np.asarray(line["coords"], dtype=int):
            if int(g[r, c]) == int(line["color"]):
                continue
            assert any(other != idx for other in cover[(int(r), int(c))]), (
                f"recovered line {line['color']} has an unexplained gap at {(int(r), int(c))}"
            )

    out = np.full_like(g, BG)
    for line in sorted(lines, key=lambda item: int(item["length"])):
        coords = np.asarray(line["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(line["color"])
    return out


def _sample_dimensions() -> tuple[int, int]:
    return random.randint(*GRID_H_RANGE), random.randint(*GRID_W_RANGE)


def _sample_lengths(count: int, max_diag: int) -> list[int]:
    upper = max(max_diag, MIN_LINE_LENGTH)
    assert upper - MIN_LINE_LENGTH + 1 >= count
    lengths = random.sample(range(MIN_LINE_LENGTH, upper + 1), count)
    random.shuffle(lengths)
    return lengths


def _sample_orientations(count: int) -> list[str]:
    distinct_min = 2 if count == 3 else 3
    distinct_max = min(4, count)
    distinct_count = random.randint(distinct_min, distinct_max)
    base = random.sample(ORIENTATIONS, distinct_count)
    orientations = base[:]
    while len(orientations) < count:
        orientations.append(random.choice(base if random.random() < 0.7 else ORIENTATIONS))
    random.shuffle(orientations)
    if orientations[0] == orientations[1]:
        for idx in range(2, len(orientations)):
            if orientations[idx] != orientations[0]:
                orientations[1], orientations[idx] = orientations[idx], orientations[1]
                break
    assert orientations[0] != orientations[1]
    return orientations


def _make_cells(start_r: int, start_c: int, orientation: str, length: int) -> list[tuple[int, int]]:
    dr, dc = STEPS[orientation]
    return [(start_r + dr * k, start_c + dc * k) for k in range(length)]


def _support_id(cells: list[tuple[int, int]], orientation: str) -> int:
    r0, c0 = cells[0]
    if orientation == "h":
        return r0
    if orientation == "v":
        return c0
    if orientation == "d1":
        return r0 - c0
    return r0 + c0


def _random_cells(H: int, W: int, orientation: str, length: int) -> list[tuple[int, int]]:
    if orientation == "h":
        return _make_cells(random.randrange(H), random.randrange(W - length + 1), orientation, length)
    if orientation == "v":
        return _make_cells(random.randrange(H - length + 1), random.randrange(W), orientation, length)
    if orientation == "d1":
        return _make_cells(random.randrange(H - length + 1), random.randrange(W - length + 1), orientation, length)
    return _make_cells(random.randrange(H - length + 1), random.randrange(length - 1, W), orientation, length)


def _crosses_interior(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    shared = set(a) & set(b)
    if len(shared) != 1:
        return False
    cell = next(iter(shared))
    return cell not in {a[0], a[-1], b[0], b[-1]}


def _crossing_count(lines: list[dict[str, object]]) -> int:
    total = 0
    for i in range(len(lines)):
        a = set(lines[i]["cells"])
        for j in range(i + 1, len(lines)):
            if a & set(lines[j]["cells"]):
                total += 1
    return total


def _compatible(cells: list[tuple[int, int]], lines: list[dict[str, object]]) -> bool:
    for line in lines:
        shared = set(cells) & set(line["cells"])
        if len(shared) > 1:
            return False
        if len(shared) == 1:
            cell = next(iter(shared))
            if cell in {cells[0], cells[-1], line["cells"][0], line["cells"][-1]}:
                return False
    return True


def _place_line(
    H: int,
    W: int,
    orientation: str,
    length: int,
    lines: list[dict[str, object]],
    used_supports: set[tuple[str, int]],
    *,
    must_cross: bool,
) -> dict[str, object] | None:
    for _ in range(LINE_PLACEMENT_ATTEMPTS):
        cells = _random_cells(H, W, orientation, length)
        support = (orientation, _support_id(cells, orientation))
        if support in used_supports:
            continue
        if not _compatible(cells, lines):
            continue
        if must_cross and not any(_crosses_interior(cells, line["cells"]) for line in lines):
            continue
        return {
            "orientation": orientation,
            "length": int(length),
            "cells": cells,
        }
    return None


def _render_output(H: int, W: int, lines: list[dict[str, object]]) -> tuple[np.ndarray, dict[tuple[int, int], list[int]]]:
    out = np.full((H, W), BG, dtype=int)
    cover = _coverage(lines)
    for line in sorted(lines, key=lambda item: int(item["length"])):
        cells = np.asarray(line["cells"], dtype=int)
        out[cells[:, 0], cells[:, 1]] = int(line["color"])
    return out, cover


def _render_input(H: int, W: int, lines: list[dict[str, object]], cover: dict[tuple[int, int], list[int]]) -> np.ndarray:
    inp = np.full((H, W), BG, dtype=int)
    mismatched_cross = False
    crossing_cells = [cell for cell, ids in cover.items() if len(ids) > 1]

    for cell, ids in cover.items():
        r, c = cell
        if len(ids) == 1:
            inp[r, c] = int(lines[ids[0]]["color"])
            continue

        top_idx = max(ids, key=lambda idx: int(lines[idx]["length"]))
        if random.random() < 0.7:
            non_top = [idx for idx in ids if idx != top_idx]
            chosen = random.choice(non_top or ids)
        else:
            chosen = random.choice(ids)
        mismatched_cross |= chosen != top_idx
        inp[r, c] = int(lines[chosen]["color"])

    if crossing_cells and not mismatched_cross:
        r, c = random.choice(crossing_cells)
        ids = cover[(r, c)]
        top_idx = max(ids, key=lambda idx: int(lines[idx]["length"]))
        non_top = [idx for idx in ids if idx != top_idx]
        chosen = random.choice(non_top)
        inp[r, c] = int(lines[chosen]["color"])

    return inp


def _visible_counts(inp: np.ndarray, lines: list[dict[str, object]]) -> list[int]:
    return [int(np.count_nonzero(inp == int(line["color"]))) for line in lines]


def _gen_scene() -> tuple[np.ndarray, np.ndarray] | None:
    H, W = _sample_dimensions()
    max_diag = min(H, W) - 1
    line_count = random.randint(*LINE_COUNT_RANGE)
    if max_diag < MIN_LINE_LENGTH or max_diag - MIN_LINE_LENGTH + 1 < line_count:
        return None

    lengths = _sample_lengths(line_count, max_diag)
    orientations = _sample_orientations(line_count)
    colors = random.sample(PALETTE, line_count)

    lines: list[dict[str, object]] = []
    used_supports: set[tuple[str, int]] = set()

    first = _place_line(H, W, orientations[0], lengths[0], lines, used_supports, must_cross=False)
    if first is None:
        return None
    lines.append(first)
    used_supports.add((first["orientation"], _support_id(first["cells"], first["orientation"])))

    second = _place_line(H, W, orientations[1], lengths[1], lines, used_supports, must_cross=True)
    if second is None:
        return None
    lines.append(second)
    used_supports.add((second["orientation"], _support_id(second["cells"], second["orientation"])))

    for orientation, length in zip(orientations[2:], lengths[2:]):
        line = _place_line(H, W, orientation, length, lines, used_supports, must_cross=False)
        if line is None:
            return None
        lines.append(line)
        used_supports.add((orientation, _support_id(line["cells"], orientation)))

    crossings = _crossing_count(lines)
    if not (1 <= crossings <= MAX_CROSSINGS_PER_SCENE):
        return None

    for line, color in zip(lines, colors):
        line["color"] = int(color)
        line["coords"] = np.array(line["cells"], dtype=int)

    out, cover = _render_output(H, W, lines)
    inp = _render_input(H, W, lines, cover)
    if min(_visible_counts(inp, lines)) < 2:
        return None
    if len(_extract_lines(inp)) != line_count:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        for _ in range(SCENE_ATTEMPTS):
            scene = _gen_scene()
            if scene is not None:
                return scene[0]
    raise RuntimeError("failed to generate valid input")
