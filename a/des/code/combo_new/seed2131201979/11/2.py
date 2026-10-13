from __future__ import annotations

import random

import numpy as np


BG = 0
ARC_MAX_SIDE = 30
COLORS = tuple(range(1, 10))

GEN_ATTEMPTS = 500
SPLIT_ATTEMPTS = 100
PLACE_ATTEMPTS = 100
MARGIN_MAX = 4
CANVAS_SIDE = 30

PAIR_COUNT_CHOICES = (2, 3, 4)
RECT_SIDE_MIN = 2
RECT_SIDE_MAX = 8
MAX_OUTPUT_WIDTH = 30

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(
    coords: list[tuple[int, int]] | set[tuple[int, int]]
) -> tuple[frozenset[tuple[int, int]], int, int, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = frozenset((r - r0, c - c0) for r, c in coords)
    return rel, r1 - r0 + 1, c1 - c0 + 1, r0, c0


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rel, obj_h, obj_w, r0, c0 = _normalize(coords)
            patch = np.full((obj_h, obj_w), BG, dtype=int)
            for rr, cc in rel:
                patch[rr, cc] = color

            objects.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "bbox": (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1),
                    "rel": rel,
                    "h": obj_h,
                    "w": obj_w,
                    "area": len(coords),
                    "patch": patch,
                }
            )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return objects


def _rect_cells(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def _pair_assemblies(a: dict, b: dict) -> list[dict]:
    if int(a["color"]) == int(b["color"]):
        return []

    total_area = int(a["area"]) + int(b["area"])
    assemblies: dict[tuple[int, int, bytes], dict] = {}

    for rect_h in range(1, total_area + 1):
        if total_area % rect_h != 0:
            continue
        rect_w = total_area // rect_h
        if rect_h > ARC_MAX_SIDE or rect_w > ARC_MAX_SIDE:
            continue
        if rect_h < int(a["h"]) or rect_w < int(a["w"]):
            continue

        rect = _rect_cells(rect_h, rect_w)
        for top in range(rect_h - int(a["h"]) + 1):
            for left in range(rect_w - int(a["w"]) + 1):
                shifted_a = {(top + r, left + c) for r, c in a["rel"]}
                comp = rect - shifted_a
                if len(comp) != int(b["area"]):
                    continue

                rel_b, b_h, b_w, _, _ = _normalize(comp)
                if b_h != int(b["h"]) or b_w != int(b["w"]) or rel_b != b["rel"]:
                    continue

                patch = np.full((rect_h, rect_w), BG, dtype=int)
                for r, c in shifted_a:
                    patch[r, c] = int(a["color"])
                for r, c in comp:
                    patch[r, c] = int(b["color"])

                key = (rect_h, rect_w, patch.tobytes())
                assemblies[key] = {
                    "area": total_area,
                    "h": rect_h,
                    "w": rect_w,
                    "patch": patch,
                }

    return [assemblies[key] for key in sorted(assemblies)]


def _pair_options(objects: list[dict]) -> dict[tuple[int, int], list[dict]]:
    options: dict[tuple[int, int], list[dict]] = {}
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            pairings = _pair_assemblies(objects[i], objects[j])
            if pairings:
                options[(i, j)] = pairings
    return options


def _enumerate_solutions(objects: list[dict]) -> list[list[tuple[int, int, dict]]]:
    n = len(objects)
    options = _pair_options(objects)
    used = [False] * n
    used_areas: set[int] = set()
    chosen: list[tuple[int, int, dict]] = []
    solutions: list[list[tuple[int, int, dict]]] = []

    def rec() -> None:
        try:
            first = next(i for i, flag in enumerate(used) if not flag)
        except StopIteration:
            solutions.append(list(chosen))
            return

        used[first] = True
        for second in range(first + 1, n):
            if used[second]:
                continue
            pairings = options.get((first, second))
            if not pairings:
                continue

            area = int(pairings[0]["area"])
            if area in used_areas:
                continue

            used[second] = True
            used_areas.add(area)
            for assembly in pairings:
                chosen.append((first, second, assembly))
                rec()
                chosen.pop()
            used_areas.remove(area)
            used[second] = False
        used[first] = False

    rec()
    return solutions


def _pack_solution(solution: list[tuple[int, int, dict]]) -> np.ndarray:
    rectangles = sorted(
        (assembly for _, _, assembly in solution),
        key=lambda item: (int(item["area"]), int(item["h"]), int(item["w"]), item["patch"].tobytes()),
    )
    out_h = max(int(rect["h"]) for rect in rectangles)
    out_w = sum(int(rect["w"]) for rect in rectangles) + len(rectangles) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    col = 0
    for rect in rectangles:
        patch = rect["patch"]
        h, w = patch.shape
        out[:h, col : col + w] = patch
        col += w + 1

    return out


def _unique_outputs(objects: list[dict]) -> tuple[list[np.ndarray], list[list[tuple[int, int, dict]]]]:
    solutions = _enumerate_solutions(objects)
    unique: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for solution in solutions:
        out = _pack_solution(solution)
        unique[(tuple(map(int, out.shape)), out.tobytes())] = out
    return list(unique.values()), solutions


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary colored fragments into rectangles, then pack the rectangles from smallest to largest area.

    Input structure:
    - The background is black.
    - The grid contains an even number of scattered monochrome 4-connected fragments.
    - Every fragment belongs to exactly one pair, and the two colors inside a pair are different.

    Rule:
    - For each pair of fragments, find a translation-only placement with no overlap whose union fills one solid rectangle exactly.
    - Keep each pixel's original fragment color while assembling that rectangle.
    - Sort the assembled rectangles by area from small to large, place them left to right with one black separator column, and align all top edges to the output's first row.

    Key insight:
    - The task has two separate steps: infer which fragments are complementary under translation, then ignore the input layout and reorder the completed rectangles by their areas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _extract_objects(g)
    assert objects and len(objects) % 2 == 0

    outputs, solutions = _unique_outputs(objects)
    assert solutions, "every fragment must be paired"
    assert len(outputs) == 1, "the complementary pairing should determine one output"
    return outputs[0]


def _sample_cut_sequence(length: int, low: int, high: int, jagged: bool) -> list[int]:
    assert low <= high
    if not jagged or low == high:
        return [random.randint(low, high)] * length

    cuts = [random.randint(low, high)]
    for _ in range(length - 1):
        delta = random.choice((-1, 0, 1))
        cuts.append(min(high, max(low, cuts[-1] + delta)))
    if len(set(cuts)) == 1:
        idx = random.randrange(length)
        bump = 1 if cuts[idx] < high else -1
        cuts[idx] += bump
    return cuts


def _piece_from_cells(
    coords: set[tuple[int, int]], color: int
) -> tuple[dict, frozenset[tuple[int, int]], int, int]:
    rel, h, w, _, _ = _normalize(coords)
    patch = np.full((h, w), BG, dtype=int)
    for r, c in rel:
        patch[r, c] = color
    piece = {"color": color, "rel": rel, "h": h, "w": w, "area": len(coords), "patch": patch}
    return piece, rel, h, w


def _split_rectangle(
    rect_h: int, rect_w: int, color_a: int, color_b: int
) -> tuple[dict, dict, np.ndarray] | None:
    total = rect_h * rect_w
    min_piece_area = max(2, total // 5)

    for _ in range(SPLIT_ATTEMPTS):
        vertical = random.random() < 0.5
        jagged = random.random() < 0.5

        if vertical:
            cuts = _sample_cut_sequence(rect_h, 1, rect_w - 1, jagged)
            cells_a = {(r, c) for r in range(rect_h) for c in range(cuts[r])}
        else:
            cuts = _sample_cut_sequence(rect_w, 1, rect_h - 1, jagged)
            cells_a = {(r, c) for c in range(rect_w) for r in range(cuts[c])}

        area_a = len(cells_a)
        area_b = total - area_a
        if min(area_a, area_b) < min_piece_area:
            continue

        full = _rect_cells(rect_h, rect_w)
        cells_b = full - cells_a
        piece_a, _, _, _ = _piece_from_cells(cells_a, color_a)
        piece_b, _, _, _ = _piece_from_cells(cells_b, color_b)

        assembled = np.full((rect_h, rect_w), BG, dtype=int)
        for r, c in cells_a:
            assembled[r, c] = color_a
        for r, c in cells_b:
            assembled[r, c] = color_b
        return piece_a, piece_b, assembled

    return None


def _sample_rectangles(pair_count: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        rects: list[tuple[int, int]] = []
        used_areas: set[int] = set()
        while len(rects) < pair_count:
            rect_h = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
            rect_w = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
            area = rect_h * rect_w
            if area in used_areas:
                continue
            rects.append((rect_h, rect_w))
            used_areas.add(area)

        out_w = sum(rect_w for _, rect_w in rects) + pair_count - 1
        if out_w <= MAX_OUTPUT_WIDTH:
            return rects
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.full((CANVAS_SIDE, CANVAS_SIDE), BG, dtype=int)
    occupied = np.zeros_like(canvas, dtype=bool)
    order = list(range(len(patches)))
    random.shuffle(order)
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        mask = patch != BG
        ph, pw = patch.shape
        valid: list[tuple[int, int]] = []

        dilated = _dilate8(occupied)
        for r0 in range(1, CANVAS_SIDE - ph):
            for c0 in range(1, CANVAS_SIDE - pw):
                if not (dilated[r0 : r0 + ph, c0 : c0 + pw] & mask).any():
                    valid.append((r0, c0))

        if not valid:
            return None

        for _ in range(min(PLACE_ATTEMPTS, len(valid))):
            r0, c0 = random.choice(valid)
            if not (_dilate8(occupied)[r0 : r0 + ph, c0 : c0 + pw] & mask).any():
                canvas[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
                occupied[r0 : r0 + ph, c0 : c0 + pw] |= mask
                break
        else:
            return None

    return canvas


def _crop_with_margin(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top = random.randint(1, min(MARGIN_MAX, int(r0)))
    left = random.randint(1, min(MARGIN_MAX, int(c0)))
    bottom = random.randint(1, min(MARGIN_MAX, int(grid.shape[0] - 1 - r1)))
    right = random.randint(1, min(MARGIN_MAX, int(grid.shape[1] - 1 - c1)))

    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1].copy()


def _expected_output(assembled_rects: list[np.ndarray]) -> np.ndarray:
    rects = sorted(assembled_rects, key=lambda patch: (int(np.count_nonzero(patch)), patch.shape[0], patch.shape[1]))
    out_h = max(patch.shape[0] for patch in rects)
    out_w = sum(patch.shape[1] for patch in rects) + len(rects) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for patch in rects:
        h, w = patch.shape
        out[:h, col : col + w] = patch
        col += w + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    rects = _sample_rectangles(pair_count)
    if rects is None:
        return None

    colors = random.sample(COLORS, 2 * pair_count)
    pieces: list[np.ndarray] = []
    assembled_rects: list[np.ndarray] = []

    for i, (rect_h, rect_w) in enumerate(rects):
        split = _split_rectangle(rect_h, rect_w, colors[2 * i], colors[2 * i + 1])
        if split is None:
            return None
        piece_a, piece_b, assembled = split
        pieces.extend([piece_a["patch"], piece_b["patch"]])
        assembled_rects.append(assembled)

    expected = _expected_output(assembled_rects)
    if max(expected.shape) > ARC_MAX_SIDE:
        return None

    scattered = _place_patches(pieces)
    if scattered is None:
        return None
    inp = _crop_with_margin(scattered)

    if inp.ndim != 2 or max(inp.shape) > ARC_MAX_SIDE:
        return None

    objects = _extract_objects(inp)
    if len(objects) != 2 * pair_count:
        return None

    outputs, solutions = _unique_outputs(objects)
    if len(solutions) != 1 or len(outputs) != 1:
        return None
    if not np.array_equal(outputs[0], expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
