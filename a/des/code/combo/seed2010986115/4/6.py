from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
FENCE_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
OPENINGS = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _u_relative_cells(height: int, width: int, opening: str) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if opening == "up":
        for r in range(height):
            cells.add((r, 0))
            cells.add((r, width - 1))
        for c in range(width):
            cells.add((height - 1, c))
        return cells
    if opening == "down":
        for r in range(height):
            cells.add((r, 0))
            cells.add((r, width - 1))
        for c in range(width):
            cells.add((0, c))
        return cells
    if opening == "left":
        for c in range(width):
            cells.add((0, c))
            cells.add((height - 1, c))
        for r in range(height):
            cells.add((r, width - 1))
        return cells
    if opening == "right":
        for c in range(width):
            cells.add((0, c))
            cells.add((height - 1, c))
        for r in range(height):
            cells.add((r, 0))
        return cells
    raise ValueError(opening)


def _detect_opening(coords: list[tuple[int, int]]) -> str | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None
    rel = {(r - r0, c - c0) for r, c in coords}
    for opening in OPENINGS:
        if rel == _u_relative_cells(height, width, opening):
            return opening
    return None


def _shift_for_pour(content: list[tuple[int, int]], fence_bbox: tuple[int, int, int, int], opening: str) -> tuple[int, int]:
    r0, c0, r1, c1 = fence_bbox
    rows = [r for r, _ in content]
    cols = [c for _, c in content]
    if opening == "up":
        return (r0 - 1) - max(rows), 0
    if opening == "down":
        return (r1 + 1) - min(rows), 0
    if opening == "left":
        return 0, (c0 - 1) - max(cols)
    if opening == "right":
        return 0, (c1 + 1) - min(cols)
    raise ValueError(opening)


def _find_fences(grid: np.ndarray) -> list[dict]:
    mask = (grid != BG) & (grid != YELLOW)
    fences: list[dict] = []
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        opening = _detect_opening(comp)
        if opening is None:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        content: list[tuple[int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(grid[r, c]) == YELLOW:
                    content.append((r, c))
        fences.append(
            {
                "color": int(next(iter(colors))),
                "bbox": (r0, c0, r1, c1),
                "opening": opening,
                "content": content,
            }
        )
    fences.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each yellow pattern out of its U-shaped fence through the missing side.
    Later fences in top-left order overwrite earlier poured content on overlap.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    fences = _find_fences(g)

    for fence in fences:
        for r, c in fence["content"]:
            out[r, c] = BG

    h, w = g.shape
    for fence in fences:
        content = fence["content"]
        if not content:
            continue
        dr, dc = _shift_for_pour(content, fence["bbox"], fence["opening"])
        for r, c in content:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = YELLOW
    return out


def _rect_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expanded(box: tuple[int, int, int, int], pad: int) -> tuple[int, int, int, int]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _sample_content(height: int, width: int) -> list[tuple[int, int]]:
    inner_h = height - 2
    inner_w = width - 2
    sub_h = random.randint(1, inner_h)
    sub_w = random.randint(1, inner_w)
    sub_r0 = random.randint(1, inner_h - sub_h + 1)
    sub_c0 = random.randint(1, inner_w - sub_w + 1)
    cells = [(r, c) for r in range(sub_r0, sub_r0 + sub_h) for c in range(sub_c0, sub_c0 + sub_w)]
    random.shuffle(cells)
    max_count = max(1, min(len(cells), (3 * len(cells) + 1) // 4))
    count = random.randint(1, max_count)
    picked = cells[:count]
    picked.sort()
    return picked


def _fence_abs_cells(spec: dict) -> set[tuple[int, int]]:
    top = spec["top"]
    left = spec["left"]
    rel = _u_relative_cells(spec["height"], spec["width"], spec["opening"])
    return {(top + r, left + c) for r, c in rel}


def _content_abs_cells(spec: dict) -> list[tuple[int, int]]:
    return [(spec["top"] + r, spec["left"] + c) for r, c in spec["content_rel"]]


def _poured_abs_cells(spec: dict) -> list[tuple[int, int]]:
    content = _content_abs_cells(spec)
    dr, dc = _shift_for_pour(content, spec["bbox"], spec["opening"])
    return [(r + dr, c + dc) for r, c in content]


def _output_union_box(spec: dict) -> tuple[int, int, int, int]:
    top = spec["top"]
    left = spec["left"]
    height = spec["height"]
    width = spec["width"]
    poured = _poured_abs_cells(spec)
    rows = [top, top + height - 1] + [r for r, _ in poured]
    cols = [left, left + width - 1] + [c for _, c in poured]
    return min(rows), min(cols), max(rows), max(cols)


def _expected_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        for r, c in _fence_abs_cells(spec):
            grid[r, c] = spec["color"]
        for r, c in _content_abs_cells(spec):
            grid[r, c] = YELLOW

    out = grid.copy()
    for spec in specs:
        for r, c in _content_abs_cells(spec):
            out[r, c] = BG
    for spec in sorted(specs, key=lambda item: (item["top"], item["left"])):
        for r, c in _poured_abs_cells(spec):
            out[r, c] = YELLOW
    return out


def _choose_openings(count: int) -> list[str]:
    dirs = list(OPENINGS)
    random.shuffle(dirs)
    if count <= len(dirs):
        return dirs[:count]
    return dirs + [random.choice(OPENINGS) for _ in range(count - len(dirs))]


def _position_candidates(shape: tuple[int, int], spec: dict) -> list[tuple[int, int]]:
    h, w = shape
    probe = dict(spec)
    probe["top"] = 0
    probe["left"] = 0
    probe["bbox"] = (0, 0, probe["height"] - 1, probe["width"] - 1)
    union_box = _output_union_box(probe)
    min_r, min_c, max_r, max_c = union_box
    row_lo = 1 - min_r
    row_hi = h - 2 - max_r
    col_lo = 1 - min_c
    col_hi = w - 2 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return []
    candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(candidates)
    return candidates


def _place_one(
    shape: tuple[int, int],
    placed: list[dict],
    template: dict,
    require_overlap: bool,
) -> dict | None:
    bbox_cells_by_spec = [(_rect_cells(*spec["bbox"]), _fence_abs_cells(spec), set(_poured_abs_cells(spec)), spec) for spec in placed]
    all_prev_bbox_cells = set().union(*(cells for cells, _fence, _poured, _spec in bbox_cells_by_spec)) if placed else set()
    all_prev_visible = set().union(
        *(fence_cells | poured_cells for _bbox_cells, fence_cells, poured_cells, _spec in bbox_cells_by_spec)
    ) if placed else set()

    for top, left in _position_candidates(shape, template):
        spec = dict(template)
        spec["top"] = top
        spec["left"] = left
        spec["bbox"] = (top, left, top + spec["height"] - 1, left + spec["width"] - 1)

        if any(_boxes_intersect(_expanded(spec["bbox"], 1), _expanded(other["bbox"], 1)) for other in placed):
            continue

        poured = set(_poured_abs_cells(spec))
        if poured & all_prev_bbox_cells:
            allowed = set()
            if require_overlap:
                allowed = all_prev_visible
            if (poured & all_prev_bbox_cells) - allowed:
                continue

        if not require_overlap:
            if poured & all_prev_visible:
                continue
            return spec

        overlap_ok = False
        overlap_cells = poured & all_prev_visible
        if overlap_cells:
            for _bbox_cells, fence_cells, poured_cells, other in bbox_cells_by_spec:
                if overlap_cells & (fence_cells | poured_cells) and (top, left) > (other["top"], other["left"]):
                    overlap_ok = True
                    break
        if overlap_ok and len(overlap_cells) <= max(1, len(poured) // 2):
            return spec
    return None


def _build_case() -> tuple[tuple[int, int], list[dict]] | None:
    shape = (random.randint(18, 28), random.randint(18, 28))
    count = random.randint(2, 4)
    colors = random.sample(FENCE_COLORS, count)
    openings = _choose_openings(count)
    overlap_index = random.randint(1, count - 1) if count >= 2 and random.random() < 0.45 else None

    placed: list[dict] = []
    for idx in range(count):
        template = {
            "color": colors[idx],
            "height": random.randint(4, 8),
            "width": random.randint(4, 8),
            "opening": openings[idx],
        }
        template["content_rel"] = _sample_content(template["height"], template["width"])
        spec = _place_one(shape, placed, template, require_overlap=idx == overlap_index)
        if spec is None:
            return None
        placed.append(spec)

    if len({spec["opening"] for spec in placed}) < min(2, len(placed)):
        return None

    if overlap_index is not None:
        ordered = sorted(placed, key=lambda item: (item["top"], item["left"]))
        visible_so_far: set[tuple[int, int]] = set()
        saw_overlap = False
        for spec in ordered:
            poured = set(_poured_abs_cells(spec))
            if poured & visible_so_far:
                saw_overlap = True
                break
            visible_so_far |= _fence_abs_cells(spec) | poured
        if not saw_overlap:
            return None

    expected = _expected_output(shape, placed)
    inp = np.full(shape, BG, dtype=int)
    for spec in placed:
        for r, c in _fence_abs_cells(spec):
            inp[r, c] = spec["color"]
        for r, c in _content_abs_cells(spec):
            inp[r, c] = YELLOW

    if not np.array_equal(main(inp), expected):
        return None
    return shape, placed


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        shape, specs = built
        inp = np.full(shape, BG, dtype=int)
        for spec in specs:
            for r, c in _fence_abs_cells(spec):
                inp[r, c] = spec["color"]
            for r, c in _content_abs_cells(spec):
                inp[r, c] = YELLOW
        return inp
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
