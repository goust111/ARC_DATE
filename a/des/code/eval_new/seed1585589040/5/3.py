import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
MAROON = 9
BORDER_COLORS = (1, 2, 3, 5, 6, 7, 8)
RECT_SIZE = 5
INNER_SIZE = 3
CANVAS_SIZE = 30
PAIR_COUNT_CHOICES = (2, 3)
GAP_RANGE = (4, 8)
PLACEMENT_ATTEMPTS = 250
MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append({"coords": arr, "bbox": (int(r0), int(c0), int(r1), int(c1))})
    return comps


def _frame_mask() -> np.ndarray:
    mask = np.zeros((RECT_SIZE, RECT_SIZE), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


FRAME_MASK = _frame_mask()


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (RECT_SIZE, RECT_SIZE):
        return False
    if len(comp["coords"]) != int(FRAME_MASK.sum()):
        return False
    local = np.zeros((RECT_SIZE, RECT_SIZE), dtype=bool)
    for r, c in comp["coords"]:
        local[int(r) - r0, int(c) - c0] = True
    return bool(np.array_equal(local, FRAME_MASK))


def _arrow_cells(bbox: tuple[int, int, int, int], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    mr = (r0 + r1) // 2
    mc = (c0 + c1) // 2
    if direction == "right":
        return [(mr, c1 + 1), (mr, c1 + 2), (mr, c1 + 3), (mr - 1, c1 + 3), (mr + 1, c1 + 3)]
    if direction == "left":
        return [(mr, c0 - 1), (mr, c0 - 2), (mr, c0 - 3), (mr - 1, c0 - 3), (mr + 1, c0 - 3)]
    if direction == "down":
        return [(r1 + 1, mc), (r1 + 2, mc), (r1 + 3, mc), (r1 + 3, mc - 1), (r1 + 3, mc + 1)]
    if direction == "up":
        return [(r0 - 1, mc), (r0 - 2, mc), (r0 - 3, mc), (r0 - 3, mc - 1), (r0 - 3, mc + 1)]
    raise ValueError(direction)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _attached_arrow_direction(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str | None:
    matches = []
    for direction in ("right", "left", "down", "up"):
        cells = _arrow_cells(bbox, direction)
        if all(_in_bounds(grid.shape, r, c) and int(grid[r, c]) == YELLOW for r, c in cells):
            matches.append(direction)
    assert len(matches) <= 1, "each source rectangle should have at most one arrow"
    return matches[0] if matches else None


def _rectangles(grid: np.ndarray) -> list[dict]:
    rects = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BLACK, YELLOW, MAROON}):
        for comp in _components(grid == color):
            if not _is_frame_component(comp):
                continue
            r0, c0, r1, c1 = comp["bbox"]
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            assert interior.shape == (INNER_SIZE, INNER_SIZE)
            assert set(map(int, np.unique(interior))).issubset({BLACK, MAROON})
            rects.append(
                {
                    "color": color,
                    "bbox": comp["bbox"],
                    "pattern": (interior == MAROON).copy(),
                    "blank": not bool(np.any(interior == MAROON)),
                    "direction": _attached_arrow_direction(grid, comp["bbox"]),
                    "center_row": (r0 + r1) // 2,
                    "center_col": (c0 + c1) // 2,
                }
            )
    return rects


def _target_for(source: dict, rects: list[dict]) -> dict:
    sr0, sc0, sr1, sc1 = source["bbox"]
    candidates = []
    for rect in rects:
        if not rect["blank"]:
            continue
        rr0, rc0, rr1, rc1 = rect["bbox"]
        if source["direction"] == "right" and rect["center_row"] == source["center_row"] and rc0 > sc1:
            candidates.append((rc0 - sc1, rect))
        elif source["direction"] == "left" and rect["center_row"] == source["center_row"] and rc1 < sc0:
            candidates.append((sc0 - rc1, rect))
        elif source["direction"] == "down" and rect["center_col"] == source["center_col"] and rr0 > sr1:
            candidates.append((rr0 - sr1, rect))
        elif source["direction"] == "up" and rect["center_col"] == source["center_col"] and rr1 < sr0:
            candidates.append((sr0 - rr1, rect))
    assert candidates, "each arrowed source must see a blank target"
    return min(candidates, key=lambda item: item[0])[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each framed 3x3 maroon pattern into the nearest blank framed target along its yellow arrow, mirroring by arrow axis.

    Input structure:
    - The background is black.
    - Foreground contains several 5x5 one-pixel-thick rectangular frames in distinct colors; each frame surrounds a 3x3 interior.
    - Some frame interiors contain a maroon pattern and some are fully black blank targets.
    - Yellow arrows sit just outside some patterned frames and point left, right, up, or down toward the nearest blank frame on the same row or column.

    Rule:
    - For each frame that has both a maroon interior pattern and an attached arrow, follow the arrow direction to the nearest blank frame aligned on that row or column.
    - Copy the source 3x3 maroon pattern into the target interior.
    - Mirror the copied pattern left-right for horizontal arrows and top-bottom for vertical arrows.
    - Remove all yellow arrow pixels in the output while leaving every frame border unchanged.

    Key insight:
    The arrow determines both which blank frame receives the pattern and which mirror axis to use during the projection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= CANVAS_SIZE

    rects = _rectangles(g)
    out = g.copy()
    out[out == YELLOW] = BLACK

    used_targets: set[tuple[int, int, int, int]] = set()
    sources = [rect for rect in rects if rect["direction"] is not None]
    assert sources, "expected at least one arrowed source rectangle"

    for source in sources:
        assert not source["blank"], "arrowed rectangles must contain a source pattern"
        target = _target_for(source, rects)
        tbbox = target["bbox"]
        assert tbbox not in used_targets, "targets should be unique"
        used_targets.add(tbbox)

        pattern = source["pattern"]
        if source["direction"] in {"left", "right"}:
            mirrored = np.fliplr(pattern)
        else:
            mirrored = np.flipud(pattern)

        tr0, tc0, tr1, tc1 = tbbox
        out[tr0 + 1 : tr1, tc0 + 1 : tc1] = np.where(mirrored, MAROON, BLACK)

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, color: int) -> None:
    r1, c1 = r0 + RECT_SIZE - 1, c0 + RECT_SIZE - 1
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color


def _draw_pattern(grid: np.ndarray, r0: int, c0: int, pattern: np.ndarray) -> None:
    interior = grid[r0 + 1 : r0 + 1 + INNER_SIZE, c0 + 1 : c0 + 1 + INNER_SIZE]
    interior[pattern] = MAROON


def _paint_arrow(grid: np.ndarray, bbox: tuple[int, int, int, int], direction: str) -> None:
    for r, c in _arrow_cells(bbox, direction):
        grid[r, c] = YELLOW


def _full_box_cells(r0: int, c0: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + RECT_SIZE) for c in range(c0, c0 + RECT_SIZE)}


def _expanded_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(shape[1], c + 2)):
                expanded.add((rr, cc))
    return expanded


def _pair_layout(direction: str, sr: int, sc: int, gap: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if direction == "right":
        return (sr, sc), (sr, sc + RECT_SIZE + gap)
    if direction == "left":
        return (sr, sc), (sr, sc - RECT_SIZE - gap)
    if direction == "down":
        return (sr, sc), (sr + RECT_SIZE + gap, sc)
    if direction == "up":
        return (sr, sc), (sr - RECT_SIZE - gap, sc)
    raise ValueError(direction)


def _placement_ranges(direction: str, gap: int) -> tuple[range, range]:
    max0 = CANVAS_SIZE - RECT_SIZE
    if direction == "right":
        return range(0, max0 + 1), range(0, CANVAS_SIZE - (2 * RECT_SIZE + gap) + 1)
    if direction == "left":
        return range(0, max0 + 1), range(RECT_SIZE + gap, max0 + 1)
    if direction == "down":
        return range(0, CANVAS_SIZE - (2 * RECT_SIZE + gap) + 1), range(0, max0 + 1)
    if direction == "up":
        return range(RECT_SIZE + gap, max0 + 1), range(0, max0 + 1)
    raise ValueError(direction)


def _all_symmetry_transforms(mask: np.ndarray) -> list[np.ndarray]:
    rot1 = np.rot90(mask, 1)
    rot2 = np.rot90(mask, 2)
    rot3 = np.rot90(mask, 3)
    trans = mask.T
    return [rot1, rot2, rot3, np.fliplr(mask), np.flipud(mask), trans, np.fliplr(trans), np.flipud(trans)]


def _sample_pattern() -> np.ndarray:
    for _ in range(300):
        mask = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
        for idx in random.sample(range(INNER_SIZE * INNER_SIZE), random.randint(3, 6)):
            mask[idx // INNER_SIZE, idx % INNER_SIZE] = True
        if all(not np.array_equal(mask, other) for other in _all_symmetry_transforms(mask)):
            return mask
    raise RuntimeError("failed to sample asymmetric 3x3 pattern")


def _place_pair(
    reserved: np.ndarray,
    direction: str,
    src_color: int,
    dst_color: int,
) -> dict | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        gap = random.randint(*GAP_RANGE)
        row_range, col_range = _placement_ranges(direction, gap)
        if not row_range or not col_range:
            return None
        sr = random.choice(tuple(row_range))
        sc = random.choice(tuple(col_range))
        (src_r, src_c), (dst_r, dst_c) = _pair_layout(direction, sr, sc, gap)
        source_bbox = (src_r, src_c, src_r + RECT_SIZE - 1, src_c + RECT_SIZE - 1)
        target_bbox = (dst_r, dst_c, dst_r + RECT_SIZE - 1, dst_c + RECT_SIZE - 1)
        raw_cells = _full_box_cells(src_r, src_c) | _full_box_cells(dst_r, dst_c) | set(_arrow_cells(source_bbox, direction))
        if any(reserved[r, c] for r, c in raw_cells):
            continue
        for r, c in _expanded_cells(raw_cells, reserved.shape):
            reserved[r, c] = True
        return {
            "direction": direction,
            "source_color": src_color,
            "target_color": dst_color,
            "source_bbox": source_bbox,
            "target_bbox": target_bbox,
            "pattern": _sample_pattern(),
        }
    return None


def _render_input(pairs: list[dict]) -> np.ndarray:
    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), BLACK, dtype=int)
    for pair in pairs:
        sr0, sc0, _, _ = pair["source_bbox"]
        tr0, tc0, _, _ = pair["target_bbox"]
        _draw_frame(grid, sr0, sc0, pair["source_color"])
        _draw_frame(grid, tr0, tc0, pair["target_color"])
        _draw_pattern(grid, sr0, sc0, pair["pattern"])
        _paint_arrow(grid, pair["source_bbox"], pair["direction"])
    return grid


def _render_output(inp: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = inp.copy()
    out[out == YELLOW] = BLACK
    for pair in pairs:
        pattern = np.fliplr(pair["pattern"]) if pair["direction"] in {"left", "right"} else np.flipud(pair["pattern"])
        tr0, tc0, tr1, tc1 = pair["target_bbox"]
        out[tr0 + 1 : tr1, tc0 + 1 : tc1] = np.where(pattern, MAROON, BLACK)
    return out


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.argwhere(inp != BLACK)
    r0, c0 = used.min(axis=0)
    r1, c1 = used.max(axis=0)
    top_margin = random.randint(0, int(r0))
    left_margin = random.randint(0, int(c0))
    bottom_margin = random.randint(0, int(CANVAS_SIZE - 1 - r1))
    right_margin = random.randint(0, int(CANVAS_SIZE - 1 - c1))
    rs = int(r0 - top_margin)
    re = int(r1 + bottom_margin + 1)
    cs = int(c0 - left_margin)
    ce = int(c1 + right_margin + 1)
    return inp[rs:re, cs:ce].copy(), out[rs:re, cs:ce].copy()


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice(PAIR_COUNT_CHOICES)
    directions = random.sample(["right", "left", "down", "up"], k=n_pairs)
    colors = random.sample(BORDER_COLORS, 2 * n_pairs)
    reserved = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    pairs = []

    for idx, direction in enumerate(directions):
        pair = _place_pair(reserved, direction, colors[2 * idx], colors[2 * idx + 1])
        if pair is None:
            return None
        pairs.append(pair)

    inp = _render_input(pairs)
    expected = _render_output(inp, pairs)
    inp, expected = _crop_pair(inp, expected)

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, actual):
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
