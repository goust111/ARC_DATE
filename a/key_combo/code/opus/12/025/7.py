from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 7, 8)
SIZE_OPTIONS = (16, 18)
OBJECT_COUNT_RANGE = (4, 6)
AREA_RANGE = (3, 12)
MAX_ATTEMPTS = 5000

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_cells(cells: set[tuple[int, int]]) -> tuple[np.ndarray, set[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox_from_cells(cells)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    shifted = {(r - r0, c - c0) for r, c in cells}
    for r, c in shifted:
        patch[r, c] = 1
    return patch, shifted


def _make_blob(area: int) -> np.ndarray:
    """Generate a small 4-connected blob with mild shape variety."""
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            base_r, base_c = random.choice(frontier)
            if random.random() < 0.35:
                # Bias toward less compact shapes so the objects are easier to see.
                candidates = sorted(
                    set(frontier),
                    key=lambda p: abs(p[0]) + abs(p[1]),
                    reverse=True,
                )
                base_r, base_c = random.choice(candidates[: max(1, len(candidates) // 2)])
            cells.add((base_r, base_c))

        if len(cells) != area:
            continue
        patch, shifted = _crop_cells(cells)
        h, w = patch.shape
        if max(h, w) > 4:
            continue
        if h * w == area and area >= 4:
            continue
        if min(h, w) == 1 and area > 5 and random.random() < 0.6:
            continue
        return patch
    raise RuntimeError("failed to generate blob")


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(cells),
                    "bbox": (r0, c0, r1, c1),
                    "area": len(cells),
                }
            )
    return out


def _center_row(bbox: tuple[int, int, int, int]) -> int:
    r0, _, r1, _ = bbox
    return (r0 + r1) // 2


def _paint_kept_only(grid: np.ndarray, kept: list[dict]) -> np.ndarray:
    out = np.zeros_like(grid)
    for obj in kept:
        rows = obj["coords"][:, 0]
        cols = obj["coords"][:, 1]
        out[rows, cols] = int(obj["color"])
    return out


def _transform(grid: np.ndarray, objects: list[dict] | None = None) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if objects is None:
        objects = _find_components(g)
    assert len(objects) >= 2, "expected at least two foreground objects"

    ranked = sorted(objects, key=lambda obj: int(obj["area"]), reverse=True)
    a_obj, b_obj = ranked[:2]
    out = _paint_kept_only(g, [a_obj, b_obj])
    protected = a_obj["coord_set"] | b_obj["coord_set"]

    h, w = g.shape
    for obj in (a_obj, b_obj):
        row = _center_row(obj["bbox"])
        color = int(obj["color"])
        for c in range(w):
            if (row, c) not in protected:
                out[row, c] = color

    mid = h // 2
    for r in range(h):
        border_color = int(a_obj["color"]) if r < mid else int(b_obj["color"])
        for c in (0, w - 1):
            if (r, c) not in protected:
                out[r, c] = border_color
    for c in range(w):
        for r in (0, h - 1):
            border_color = int(a_obj["color"]) if r < mid else int(b_obj["color"])
            if (r, c) not in protected:
                out[r, c] = border_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the two largest objects, draw their horizontal guide lines, and split the frame by area rank.

    Input structure:
    - The input is a black square grid of even height with 4-6 monochromatic 4-connected objects.
    - Each object uses a different non-black color, has a distinct area, and is separated from the others.
    - The two largest objects have bbox-center rows in opposite half-grids, but the larger one may be above or below.

    Rule:
    - Measure every foreground object's 4-connected area and keep only the largest object A and second-largest object B.
    - Draw a full-width horizontal line through A's bbox center row in A's color and through B's bbox center row in B's color, skipping the original pixels of A and B.
    - Draw a 1-pixel border whose rows in the upper half use A's color and whose rows in the lower half use B's color; when border cells overlap line cells, the border wins, but original A/B object pixels stay unchanged.

    Key insight:
    - Position controls where the two horizontal lines appear, while area rank alone decides which kept color owns the top border half and which owns the bottom half.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _transform(g)


def _center_row_in_half(top: int, patch_h: int, size: int, half: str) -> bool:
    row = top + (patch_h - 1) // 2
    return row < size // 2 if half == "top" else row >= size // 2


def _slot_origins(size: int) -> list[int]:
    return [0, 6, 12] if size == 16 else [0, 7, 14]


def _slot_positions(
    size: int,
    patch: np.ndarray,
    slot_r: int,
    slot_c: int,
    *,
    half: str | None = None,
    border_side: str | None = None,
) -> list[tuple[int, int]]:
    cell = 4
    h, w = patch.shape
    candidates = []
    for dr in range(cell - h + 1):
        top = slot_r + dr
        if half is not None and not _center_row_in_half(top, h, size, half):
            continue
        if border_side == "bottom":
            left_choices = range(cell - w + 1)
            if top + h != size:
                continue
        else:
            left_choices = range(cell - w + 1)
        for dc in left_choices:
            left = slot_c + dc
            if border_side == "left" and left != 0:
                continue
            if border_side == "right" and left + w != size:
                continue
            candidates.append((top, left))
    return candidates


def _place_patch(
    canvas: np.ndarray,
    patch: np.ndarray,
    color: int,
    slot: tuple[int, int],
    *,
    half: str | None = None,
    border_side: str | None = None,
) -> tuple[int, int] | None:
    size = canvas.shape[0]
    top_lefts = _slot_positions(size, patch, slot[0], slot[1], half=half, border_side=border_side)
    if not top_lefts:
        return None
    top, left = random.choice(top_lefts)
    rows, cols = np.argwhere(patch != 0).T
    canvas[top + rows, left + cols] = color
    return top, left


def _pairwise_separated(objects: list[dict]) -> bool:
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a = objects[i]["coord_set"]
            b = objects[j]["coord_set"]
            for r, c in a:
                for dr, dc in DIR8:
                    if (r + dr, c + dc) in b:
                        return False
                if (r, c) in b:
                    return False
    return True


def _validate_input(grid: np.ndarray) -> list[dict] | None:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return None
    h, w = grid.shape
    if h != w or h % 2 != 0 or h > 30:
        return None
    objects = _find_components(grid)
    if not (OBJECT_COUNT_RANGE[0] <= len(objects) <= OBJECT_COUNT_RANGE[1]):
        return None
    colors = [int(obj["color"]) for obj in objects]
    if len(colors) != len(set(colors)):
        return None
    areas = [int(obj["area"]) for obj in objects]
    if len(areas) != len(set(areas)):
        return None
    if not all(AREA_RANGE[0] <= area <= AREA_RANGE[1] for area in areas):
        return None
    if not _pairwise_separated(objects):
        return None
    ranked = sorted(objects, key=lambda obj: int(obj["area"]), reverse=True)
    top_row = _center_row(ranked[0]["bbox"])
    bottom_row = _center_row(ranked[1]["bbox"])
    if (top_row < h // 2) == (bottom_row < h // 2):
        return None
    return objects


def _check_output_semantics(inp: np.ndarray, out: np.ndarray, objects: list[dict]) -> bool:
    if out.shape != inp.shape:
        return False
    ranked = sorted(objects, key=lambda obj: int(obj["area"]), reverse=True)
    a_obj, b_obj = ranked[:2]
    protected = a_obj["coord_set"] | b_obj["coord_set"]
    h, w = inp.shape
    mid = h // 2

    others = ranked[2:]
    for obj in others:
        rows = obj["coords"][:, 0]
        cols = obj["coords"][:, 1]
        if np.any(out[rows, cols] != BG):
            return False

    for obj in (a_obj, b_obj):
        rows = obj["coords"][:, 0]
        cols = obj["coords"][:, 1]
        if np.any(out[rows, cols] != int(obj["color"])):
            return False

    a_row = _center_row(a_obj["bbox"])
    b_row = _center_row(b_obj["bbox"])
    for c in range(w):
        if (a_row, c) not in protected and a_row not in (0, h - 1):
            if int(out[a_row, c]) != int(a_obj["color"]):
                return False
        if (b_row, c) not in protected and b_row not in (0, h - 1):
            if int(out[b_row, c]) != int(b_obj["color"]):
                return False

    for r in range(h):
        border_color = int(a_obj["color"]) if r < mid else int(b_obj["color"])
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if (r, c) in protected:
                continue
            if int(out[r, c]) != border_color:
                return False
    return True


def _generate_once() -> np.ndarray | None:
    size = random.choice(SIZE_OPTIONS)
    count = random.randint(*OBJECT_COUNT_RANGE)
    colors = random.sample(PALETTE, count)
    areas = random.sample(range(AREA_RANGE[0], AREA_RANGE[1] + 1), count)
    areas.sort(reverse=True)

    try:
        patches = [_make_blob(area) for area in areas]
    except RuntimeError:
        return None
    canvas = np.zeros((size, size), dtype=int)
    origins = _slot_origins(size)
    slots = [(r, c) for r in origins for c in origins]
    used_slots: set[tuple[int, int]] = set()

    larger_in_top = random.random() < 0.5
    touch_border_idx = random.choice([None, 0, 1]) if random.random() < 0.55 else None

    for idx, half in [
        (0, "top" if larger_in_top else "bottom"),
        (1, "bottom" if larger_in_top else "top"),
    ]:
        border_side = None
        if half == "top":
            row_choices = [origins[1], origins[0]] if touch_border_idx == idx else [origins[0], origins[1]]
        else:
            row_choices = [origins[2]]
        col_choices = [origins[0], origins[2], origins[1]]
        if touch_border_idx == idx:
            border_side = random.choice(["left", "right"])
            col_choices = [origins[0], origins[2]] if border_side in {"left", "right"} else col_choices

        slot_choices = [
            (r0, c0)
            for r0 in row_choices
            for c0 in col_choices
            if (r0, c0) not in used_slots
        ]
        random.shuffle(slot_choices)
        loc = None
        for slot in slot_choices:
            loc = _place_patch(
                canvas,
                patches[idx],
                colors[idx],
                slot,
                half=half,
                border_side=border_side,
            )
            if loc is not None:
                used_slots.add(slot)
                break
        if loc is None:
            return None

    tail = list(range(2, count))
    random.shuffle(tail)
    for idx in tail:
        slot_choices = [slot for slot in slots if slot not in used_slots]
        random.shuffle(slot_choices)
        loc = None
        for slot in slot_choices:
            loc = _place_patch(canvas, patches[idx], colors[idx], slot, half=None, border_side=None)
            if loc is not None:
                used_slots.add(slot)
                break
        if loc is None:
            return None

    objects = _validate_input(canvas)
    if objects is None:
        return None
    out = main(canvas)
    if not _check_output_semantics(canvas, out, objects):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_once()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
