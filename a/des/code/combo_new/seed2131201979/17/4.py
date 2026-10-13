from __future__ import annotations

from collections import deque
import itertools
import random

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _count_holes(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    holes = 0
    for comp in _components_mask(~mask):
        rows = comp[:, 0]
        cols = comp[:, 1]
        touches_edge = bool(
            (rows == 0).any() or (rows == h - 1).any() or (cols == 0).any() or (cols == w - 1).any()
        )
        if not touches_edge:
            holes += 1
    return holes


def _object_components(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    out: list[tuple[int, np.ndarray]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for comp in _components_mask(g == color):
            out.append((int(color), comp))
    return out


def _draw_rings(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    bg: int,
    layers: int,
) -> None:
    r0, c0, r1, c1 = bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    for layer in range(int(layers)):
        inset = 2 * int(layer)
        top = r0 + inset
        left = c0 + inset
        bottom = r1 - inset
        right = c1 - inset
        if top > bottom or left > right:
            break
        out[top, left : right + 1] = int(color)
        out[bottom, left : right + 1] = int(color)
        out[top : bottom + 1, left] = int(color)
        out[top : bottom + 1, right] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each foreground object by concentric rectangular rings whose count is determined by the object's hole count.

    Input structure:
    - The grid has one dominant background color and several separated foreground objects.
    - Each object is a single 4-connected monochrome component with a non-background color.
    - Objects may be solid or may contain one or more enclosed background holes, and their bounding boxes do not overlap.

    Rule:
    - For each foreground object, count the number N of 4-connected background regions that are fully enclosed by that object's color and do not touch the object's bounding-box border.
    - Clear the object's bounding box to background, then draw N+1 concentric rectangular border rings of the same color inside that box.
    - The outer ring coincides with the bounding-box border, each further ring is inset by two cells so adjacent rings are separated by one background cell, and drawing stops once no inner rectangle remains.
    - Pixels outside every object bounding box stay unchanged.

    Key insight:
    - The original silhouette is discarded; only the object's color and its number of enclosed holes matter, and the hole count controls how many concentric rings appear.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color, comp in _object_components(g, bg):
        r0, c0, r1, c1 = _bbox(comp)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local_mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
        hole_count = _count_holes(local_mask)
        _draw_rings(out, (r0, c0, r1, c1), color, bg, hole_count + 1)

    return out


# --------------- generator ---------------

GRID_LIMIT = 30
PALETTE = tuple(range(10))
MAX_GENERATE_ATTEMPTS = 500
MAX_OBJECTS = 4
OUTER_SEGMENT_BASE = 1
MAX_LAYOUT_SEGMENTS = 3


def _hole_slots(height: int, width: int, margin: int) -> list[tuple[int, int]]:
    if height < 2 * margin + 1 or width < 2 * margin + 1:
        return []
    return [
        (r, c)
        for r in range(margin, height - margin, 2)
        for c in range(margin, width - margin, 2)
    ]


def _dimension_candidates(
    hole_count: int,
    branch: str,
    hole_margin: int,
    max_side: int,
) -> list[tuple[int, int]]:
    min_side = 3 if hole_count == 0 else 2 * hole_margin + 1
    candidates: list[tuple[int, int]] = []
    for height in range(min_side, max_side + 1):
        for width in range(min_side, max_side + 1):
            if hole_count > len(_hole_slots(height, width, hole_margin)):
                continue
            if hole_count > 0 and branch == "full" and min(height, width) < 4 * hole_count + 1:
                continue
            if hole_count > 0 and branch == "truncated" and min(height, width) >= 4 * hole_count + 1:
                continue
            if height * width < max(5, hole_count + 4):
                continue
            candidates.append((height, width))
    return candidates


def _make_object_mask(
    height: int,
    width: int,
    hole_count: int,
    hole_margin: int,
    allow_notches: bool,
) -> np.ndarray | None:
    slots = _hole_slots(height, width, hole_margin)
    if hole_count > len(slots):
        return None

    for _ in range(120):
        mask = np.ones((height, width), dtype=bool)
        if hole_count > 0:
            for r, c in random.sample(slots, hole_count):
                mask[r, c] = False

        if allow_notches:
            sides = random.sample(("top", "bottom", "left", "right"), random.randint(0, 3))
            for side in sides:
                if side in ("top", "bottom"):
                    if width < 2:
                        continue
                    span = random.randint(1, width - 1)
                    start = random.randint(0, width - span)
                    row = 0 if side == "top" else height - 1
                    mask[row, start : start + span] = False
                else:
                    if height < 2:
                        continue
                    span = random.randint(1, height - 1)
                    start = random.randint(0, height - span)
                    col = 0 if side == "left" else width - 1
                    mask[start : start + span, col] = False

        if mask.sum() == 0:
            continue
        if len(_components_mask(mask)) != 1:
            continue
        if _count_holes(mask) != hole_count:
            continue
        return mask

    return None


def _sample_object_specs(num_objects: int) -> list[dict] | None:
    bg = random.choice(PALETTE)
    object_colors = random.sample([c for c in PALETTE if c != bg], num_objects)
    side_cap = 18 if num_objects == 1 else 14 if num_objects == 2 else 10

    branches = ["full", "truncated"] if num_objects >= 2 else [random.choice(["full", "truncated"])]
    while len(branches) < num_objects:
        branches.append(random.choice(["full", "truncated"]))
    random.shuffle(branches)

    specs: list[dict] = []
    for idx in range(num_objects):
        branch = branches[idx]
        if branch == "full":
            max_full_holes = min(4, max(0, (side_cap - 1) // 4))
            if max_full_holes == 0:
                hole_count = 0
            elif random.random() < 0.25:
                hole_count = 0
            else:
                hole_count = random.randint(1, max_full_holes)
            hole_margin = 2 if hole_count > 0 else 1
            allow_notches = hole_margin >= 2
        else:
            hole_count = random.randint(1, 4)
            hole_margin = random.choice([1, 2])
            allow_notches = hole_margin >= 2

        candidates = _dimension_candidates(hole_count, branch, hole_margin, side_cap)
        if not candidates:
            return None
        height, width = random.choice(candidates)
        mask = _make_object_mask(height, width, hole_count, hole_margin, allow_notches)
        if mask is None:
            return None

        specs.append(
            {
                "branch": branch,
                "color": int(object_colors[idx]),
                "holes": int(hole_count),
                "mask": mask,
                "height": int(height),
                "width": int(width),
            }
        )

    return [{"bg": bg}] + specs


def _layout_candidates(num_objects: int) -> list[tuple[int, int]]:
    layouts = {(1, num_objects), (num_objects, 1)}
    if num_objects >= 2:
        layouts.add((2, (num_objects + 1) // 2))
        layouts.add(((num_objects + 1) // 2, 2))
    return sorted(layouts)


def _composition(total: int, parts: int) -> list[int]:
    if parts <= 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    prev = -1
    out = []
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _choose_packing(specs: list[dict]) -> dict | None:
    objects = specs[1:]
    num_objects = len(objects)
    best: list[dict] = []
    total_area = sum(int(obj["mask"].sum()) for obj in objects)
    max_area = max(int(obj["mask"].sum()) for obj in objects)

    for rows, cols in _layout_candidates(num_objects):
        if rows * cols < num_objects:
            continue
        for perm in itertools.permutations(range(num_objects)):
            row_heights = [0] * rows
            col_widths = [0] * cols
            for slot_idx, obj_idx in enumerate(perm):
                rr = slot_idx // cols
                cc = slot_idx % cols
                row_heights[rr] = max(row_heights[rr], int(objects[obj_idx]["height"]))
                col_widths[cc] = max(col_widths[cc], int(objects[obj_idx]["width"]))

            base_h = sum(row_heights) + OUTER_SEGMENT_BASE * (rows + 1)
            base_w = sum(col_widths) + OUTER_SEGMENT_BASE * (cols + 1)
            if base_h > GRID_LIMIT or base_w > GRID_LIMIT:
                continue

            size_candidates = [
                (hh, ww)
                for hh in range(base_h, GRID_LIMIT + 1)
                for ww in range(base_w, GRID_LIMIT + 1)
                if hh * ww > total_area + max_area
            ]
            if not size_candidates:
                continue
            size_candidates.sort(key=lambda t: (t[0] * t[1], t[0] + t[1]))
            for hh, ww in size_candidates[:40]:
                best.append(
                    {
                        "rows": rows,
                        "cols": cols,
                        "perm": perm,
                        "row_heights": row_heights,
                        "col_widths": col_widths,
                        "height": hh,
                        "width": ww,
                    }
                )

    if not best:
        return None
    return random.choice(best)


def _place_objects(specs: list[dict], packing: dict) -> tuple[np.ndarray, list[dict]]:
    bg = int(specs[0]["bg"])
    objects = specs[1:]
    rows = int(packing["rows"])
    cols = int(packing["cols"])
    row_heights = [int(v) for v in packing["row_heights"]]
    col_widths = [int(v) for v in packing["col_widths"]]
    grid_h = int(packing["height"])
    grid_w = int(packing["width"])

    extra_h = grid_h - (sum(row_heights) + OUTER_SEGMENT_BASE * (rows + 1))
    extra_w = grid_w - (sum(col_widths) + OUTER_SEGMENT_BASE * (cols + 1))
    v_segments = [OUTER_SEGMENT_BASE] * (rows + 1)
    h_segments = [OUTER_SEGMENT_BASE] * (cols + 1)
    for i, extra in enumerate(_composition(extra_h, rows + 1)):
        v_segments[i] += extra
    for i, extra in enumerate(_composition(extra_w, cols + 1)):
        h_segments[i] += extra

    row_starts: list[int] = []
    cur = v_segments[0]
    for rr in range(rows):
        row_starts.append(cur)
        cur += row_heights[rr] + v_segments[rr + 1]

    col_starts: list[int] = []
    cur = h_segments[0]
    for cc in range(cols):
        col_starts.append(cur)
        cur += col_widths[cc] + h_segments[cc + 1]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    placed: list[dict] = []
    for slot_idx, obj_idx in enumerate(packing["perm"]):
        rr = slot_idx // cols
        cc = slot_idx % cols
        obj = objects[obj_idx]
        mask = obj["mask"]
        obj_h, obj_w = mask.shape
        top = row_starts[rr] + random.randint(0, row_heights[rr] - obj_h)
        left = col_starts[cc] + random.randint(0, col_widths[cc] - obj_w)
        grid[top : top + obj_h, left : left + obj_w][mask] = int(obj["color"])
        placed.append(
            {
                "color": int(obj["color"]),
                "holes": int(obj["holes"]),
                "bbox": (top, left, top + obj_h - 1, left + obj_w - 1),
                "mask": mask,
            }
        )

    return grid, placed


def _expected_output(grid: np.ndarray, bg: int, placed: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for obj in placed:
        _draw_rings(out, obj["bbox"], int(obj["color"]), int(bg), int(obj["holes"]) + 1)
    return out


def _is_valid_input(grid: np.ndarray, bg: int, placed: list[dict], expected: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > GRID_LIMIT:
        return False
    if _get_bg(grid) != int(bg):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    seen_bboxes = []
    for obj in placed:
        color = int(obj["color"])
        r0, c0, r1, c1 = obj["bbox"]
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            return False
        local = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        if _count_holes(local) != int(obj["holes"]):
            return False
        for pr0, pc0, pr1, pc1 in seen_bboxes:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                return False
        seen_bboxes.append(obj["bbox"])

    vals, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[np.where(vals == bg)[0][0]])
    other_counts = [int(c) for v, c in zip(vals, counts) if int(v) != int(bg)]
    return bg_count > max(other_counts, default=0)


def _generate_attempt() -> np.ndarray | None:
    num_objects = random.randint(1, MAX_OBJECTS)
    specs = _sample_object_specs(num_objects)
    if specs is None:
        return None

    packing = _choose_packing(specs)
    if packing is None:
        return None

    grid, placed = _place_objects(specs, packing)
    bg = int(specs[0]["bg"])
    expected = _expected_output(grid, bg, placed)
    if not _is_valid_input(grid, bg, placed, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
