from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6

SCATTER_COLORS = (BLUE, YELLOW, GREY, MAGENTA)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)


def _neighbors(shape: tuple[int, int], r: int, c: int, connectivity: int) -> Iterable[tuple[int, int]]:
    h, w = shape
    deltas = DIR4 if connectivity == 4 else DIR8
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
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
                for nr, nc in _neighbors(mask.shape, rr, cc, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expand_bbox(bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _is_domino(coords: np.ndarray) -> bool:
    if int(coords.shape[0]) != 2:
        return False
    (r0, c0), (r1, c1) = coords.tolist()
    return abs(r0 - r1) + abs(c0 - c1) == 1


def _paint_frame(out: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = frame_bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                if out[r, c] == BLACK:
                    out[r, c] = GREEN


def _captured_colors(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> set[int]:
    r0, c0, r1, c1 = frame_bbox
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = (crop != BLACK) & (crop != RED) & (crop != GREEN)
    return {int(v) for v in crop[mask]}


def _draw_marker(out: np.ndarray, frame_bbox: tuple[int, int, int, int], height: int) -> None:
    if height <= 0:
        return
    h, _w = out.shape
    _r0, c0, r1, c1 = frame_bbox
    top = r1 + 1
    bottom = min(h - 1, r1 + height)
    if top > bottom:
        return
    out[top : bottom + 1, c0 : c1 + 1] = YELLOW


def _parse_red_objects(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]]]:
    red_comps = _components_mask(grid == RED, connectivity=4)
    dominoes: list[dict] = []
    singles: list[tuple[int, int]] = []
    for comp in red_comps:
        area = int(comp.shape[0])
        if area == 1:
            singles.append(tuple(int(v) for v in comp[0]))
            continue
        if _is_domino(comp):
            bbox = _bbox_coords(comp)
            dominoes.append(
                {
                    "coords": {(int(r), int(c)) for r, c in comp},
                    "bbox": bbox,
                    "frame_bbox": _expand_bbox(bbox, grid.shape, pad=1),
                }
            )
            continue
        raise AssertionError("red objects must be isolated single pixels or 2-cell dominoes")
    dominoes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    singles.sort()
    return dominoes, singles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Outline red dominoes in green, count the captured color diversity in each framed box, and place a yellow height marker below.

    Input structure:
    - The grid is mostly black background.
    - Red pixels form 4-connected objects that are either single isolated pixels or 2-cell horizontal/vertical dominoes.
    - Additional colored single pixels are scattered away from the red-object classification signal.

    Rule:
    - Identify every red 4-connected component and separate single pixels from dominoes.
    - For each domino, expand its bounding box by one cell in every direction, clipped to the grid, and paint the perimeter green only on currently black cells.
    - Within that expanded framed box, count the distinct colors of all non-black, non-red, non-green pixels; call that diversity D.
    - Draw a yellow rectangle immediately below the frame, with the same width as the frame and height D, clipping at the bottom if needed.
    - Erase every red single pixel to black while keeping red dominoes and captured colored dots unchanged.

    Key insight:
    - The green outline defines which local colored dots belong to each domino, and the number of distinct captured colors determines the marker height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    dominoes, singles = _parse_red_objects(g)
    out = g.copy()

    for domino in dominoes:
        frame_bbox = domino["frame_bbox"]
        _paint_frame(out, frame_bbox)
        diversity = len(_captured_colors(out, frame_bbox))
        _draw_marker(out, frame_bbox, diversity)

    for r, c in singles:
        out[r, c] = BLACK
    return out


def _touches_edge(frame_bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    h, w = shape
    r0, c0, r1, c1 = frame_bbox
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _domino_cells(r: int, c: int, orient: str) -> list[tuple[int, int]]:
    if orient == "H":
        return [(r, c), (r, c + 1)]
    return [(r, c), (r + 1, c)]


def _rect_intersects(mask: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return bool(mask[r0 : r1 + 1, c0 : c1 + 1].any())


def _mark_rect(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _frame_domain(shape: tuple[int, int], coords: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    bbox = _bbox_coords(np.array(coords, dtype=int))
    frame_bbox = _expand_bbox(bbox, shape, pad=1)
    coord_set = set(coords)
    r0, c0, r1, c1 = frame_bbox
    domain = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in coord_set
    ]
    return frame_bbox, domain


def _choose_isolated_cells(
    rng: np.random.Generator, cells: list[tuple[int, int]], count: int
) -> list[tuple[int, int]] | None:
    cells = list(cells)
    rng.shuffle(cells)
    neighbors = {
        cell: {
            other
            for other in cells
            if max(abs(cell[0] - other[0]), abs(cell[1] - other[1])) <= 1 and other != cell
        }
        for cell in cells
    }

    best: list[tuple[int, int]] = []

    def backtrack(start: int, chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        nonlocal best
        if len(chosen) == count:
            return list(chosen)
        if len(chosen) > len(best):
            best = list(chosen)
        remaining = len(cells) - start
        if len(chosen) + remaining < count:
            return None
        for idx in range(start, len(cells)):
            cell = cells[idx]
            if any(other in neighbors[cell] for other in chosen):
                continue
            chosen.append(cell)
            found = backtrack(idx + 1, chosen)
            if found is not None:
                return found
            chosen.pop()
        return None

    return backtrack(0, [])


def _scatter_palette_for_diversity(
    rng: np.random.Generator, diversity: int, force_repeat: bool
) -> list[int]:
    distinct = list(rng.choice(SCATTER_COLORS, size=diversity, replace=False))
    if diversity == 4:
        colors = distinct
    else:
        low = diversity + 1 if force_repeat else diversity
        count = int(rng.integers(low, 5))
        colors = list(distinct)
        while len(colors) < count:
            colors.append(int(rng.choice(distinct)))
        rng.shuffle(colors)
    return [int(c) for c in colors]


def _candidate_dominoes(
    shape: tuple[int, int],
    diversity: int,
    reserved: np.ndarray,
) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []
    for orient in ("H", "V"):
        row_stop = h if orient == "H" else h - 1
        col_stop = w - 1 if orient == "H" else w
        for r in range(row_stop):
            for c in range(col_stop):
                coords = _domino_cells(r, c, orient)
                frame_bbox, domain = _frame_domain(shape, coords)
                marker_bottom = frame_bbox[2] + diversity
                if marker_bottom >= h:
                    continue
                usable_rect = (frame_bbox[0], frame_bbox[1], marker_bottom, frame_bbox[3])
                padded_rect = _expand_bbox(usable_rect, shape, pad=1)
                if _rect_intersects(reserved, padded_rect):
                    continue
                if len(domain) < 4:
                    continue
                candidates.append(
                    {
                        "orient": orient,
                        "coords": coords,
                        "bbox": _bbox_coords(np.array(coords, dtype=int)),
                        "frame_bbox": frame_bbox,
                        "domain": domain,
                        "usable_rect": usable_rect,
                        "padded_rect": padded_rect,
                    }
                )
    return candidates


def _position_is_isolated(grid: np.ndarray, pos: tuple[int, int], color: int | None = None) -> bool:
    r, c = pos
    if grid[r, c] != BLACK:
        return False
    for nr, nc in _neighbors(grid.shape, r, c, 8):
        if grid[nr, nc] != BLACK:
            if color is None or grid[nr, nc] == color:
                return False
    return True


def _place_extra_scatter(
    rng: np.random.Generator,
    grid: np.ndarray,
    forbidden: np.ndarray,
    count: int,
) -> list[tuple[int, int, int]]:
    h, w = grid.shape
    placed: list[tuple[int, int, int]] = []
    for _ in range(count):
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if not forbidden[r, c] and _position_is_isolated(grid, (r, c))
        ]
        if not candidates:
            break
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        color = int(rng.choice(SCATTER_COLORS))
        grid[r, c] = color
        placed.append((r, c, color))
    return placed


def _place_red_singles(
    rng: np.random.Generator,
    grid: np.ndarray,
    forbidden: np.ndarray,
    count: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    placed: list[tuple[int, int]] = []
    for _ in range(count):
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if not forbidden[r, c] and _position_is_isolated(grid, (r, c), color=RED)
        ]
        if not candidates:
            break
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        grid[r, c] = RED
        placed.append((r, c))
    return placed


def _render_from_metadata(
    grid: np.ndarray,
    dominoes: list[dict],
    red_singles: list[tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for domino in sorted(dominoes, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        _paint_frame(out, domino["frame_bbox"])
        diversity = len(set(domino["captured_colors"]))
        _draw_marker(out, domino["frame_bbox"], diversity)
    for r, c in red_singles:
        out[r, c] = BLACK
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _validate_generated(
    grid: np.ndarray,
    dominoes: list[dict],
    red_singles: list[tuple[int, int]],
    extras: list[tuple[int, int, int]],
) -> bool:
    parsed_dominoes, parsed_singles = _parse_red_objects(grid)
    if len(parsed_dominoes) != len(dominoes) or sorted(red_singles) != parsed_singles:
        return False

    parsed_by_bbox = {dom["bbox"]: dom for dom in parsed_dominoes}
    target_diversities = []
    repeated_capture = False
    edge_frame_present = False

    for domino in dominoes:
        if domino["bbox"] not in parsed_by_bbox:
            return False
        parsed = parsed_by_bbox[domino["bbox"]]
        if parsed["frame_bbox"] != domino["frame_bbox"]:
            return False
        for (r, c), color in domino["captured"].items():
            if int(grid[r, c]) != int(color):
                return False
        diversity = len(set(domino["captured_colors"]))
        target_diversities.append(diversity)
        repeated_capture |= len(domino["captured_colors"]) > diversity
        edge_frame_present |= _touches_edge(domino["frame_bbox"], grid.shape)

    if len(set(target_diversities)) != len(target_diversities):
        return False
    if not repeated_capture:
        return False
    if not extras:
        return False

    frame_boxes = [dom["frame_bbox"] for dom in dominoes]
    for i in range(len(frame_boxes)):
        for j in range(i + 1, len(frame_boxes)):
            if _boxes_overlap(frame_boxes[i], frame_boxes[j]):
                return False

    for r, c in red_singles:
        if any(box[0] <= r <= box[2] and box[1] <= c <= box[3] for box in frame_boxes):
            return False

    expected = _render_from_metadata(grid, dominoes, red_singles)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    for domino in parsed_dominoes:
        diversity = len(_captured_colors(grid, domino["frame_bbox"]))
        r0, c0, r1, c1 = domino["frame_bbox"]
        marker = actual[r1 + 1 :, c0 : c1 + 1] if r1 + 1 < actual.shape[0] else np.zeros((0, c1 - c0 + 1), dtype=int)
        marker_rows = 0
        for row in marker:
            if np.all(row == YELLOW):
                marker_rows += 1
            else:
                break
        if marker_rows != diversity:
            return False

    return True


def _generate_attempt(rng: np.random.Generator) -> tuple[np.ndarray, list[dict], list[tuple[int, int]], list[tuple[int, int, int]]] | None:
    h = int(rng.integers(14, 21))
    w = int(rng.integers(14, 21))
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    domino_count = int(rng.choice(np.array([2, 3, 4])))
    diversities = list(rng.choice(np.array([1, 2, 3, 4]), size=domino_count, replace=False))
    rng.shuffle(diversities)
    repeat_index_choices = [idx for idx, d in enumerate(diversities) if d < 4]
    repeat_index = int(rng.choice(np.array(repeat_index_choices)))

    dominoes: list[dict] = []
    for idx, diversity in enumerate(diversities):
        candidates = _candidate_dominoes(grid.shape, diversity, reserved)
        if not candidates:
            return None
        prefer_edge = rng.random() < 0.35
        if prefer_edge:
            edge_candidates = [cand for cand in candidates if _touches_edge(cand["frame_bbox"], grid.shape)]
            if edge_candidates:
                candidates = edge_candidates
        chosen = candidates[int(rng.integers(0, len(candidates)))]
        captured_colors = _scatter_palette_for_diversity(rng, diversity, force_repeat=(idx == repeat_index))
        captured_count = len(captured_colors)
        captured_cells = _choose_isolated_cells(rng, chosen["domain"], captured_count)
        if captured_cells is None:
            return None

        for r, c in chosen["coords"]:
            grid[r, c] = RED
        for (r, c), color in zip(captured_cells, captured_colors):
            grid[r, c] = int(color)

        domino = {
            "orient": chosen["orient"],
            "coords": chosen["coords"],
            "bbox": chosen["bbox"],
            "frame_bbox": chosen["frame_bbox"],
            "captured": {cell: color for cell, color in zip(captured_cells, captured_colors)},
            "captured_colors": list(captured_colors),
        }
        dominoes.append(domino)
        _mark_rect(reserved, chosen["padded_rect"])

    forbidden = reserved.copy()
    for domino in dominoes:
        _mark_rect(forbidden, domino["frame_bbox"])

    extras = _place_extra_scatter(rng, grid, forbidden, count=int(rng.integers(1, 4)))
    if not extras:
        return None
    red_singles = _place_red_singles(rng, grid, forbidden, count=int(rng.integers(1, 5)))
    if not red_singles:
        return None

    if not _validate_generated(grid, dominoes, red_singles, extras):
        return None
    return grid, dominoes, red_singles, extras


def generate_input() -> np.ndarray:
    for _ in range(500):
        sample = _generate_attempt(np.random.default_rng())
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid input for puzzle 3")
