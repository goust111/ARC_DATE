from __future__ import annotations

import random
from collections import deque

import numpy as np


BG_COLOR = 8  # Azure
FG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 400


def _paste_mask(dst: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = mask.shape
    block = dst[top : top + h, left : left + w]
    block[mask] = color


def _find_components(grid: np.ndarray, bg: int = BG_COLOR) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            obj_h = r1 - r0 + 1
            obj_w = c1 - c0 + 1
            mask = np.zeros((obj_h, obj_w), dtype=bool)
            for rr, cc in cells:
                mask[rr - r0, cc - c0] = True

            objects.append(
                {
                    "color": color,
                    "area": len(cells),
                    "bbox": (r0, c0, r1, c1),
                    "h": obj_h,
                    "w": obj_w,
                    "mask": mask,
                }
            )

    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Stack wide objects downward on the left and tall objects across the bottom."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _find_components(g, bg=BG_COLOR)
    assert objects, "expected at least one foreground object"

    horizontal: list[dict] = []
    vertical: list[dict] = []
    for obj in objects:
        h = int(obj["h"])
        w = int(obj["w"])
        assert h != w, "objects must have non-square bounding boxes"
        if w > h:
            horizontal.append(obj)
        else:
            vertical.append(obj)

    assert horizontal or vertical

    horizontal.sort(key=lambda obj: (-int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1])))
    vertical.sort(key=lambda obj: (-int(obj["area"]), int(obj["bbox"][0]), int(obj["bbox"][1])))

    left_w = max((int(obj["w"]) for obj in horizontal), default=0)
    stack_h = sum(int(obj["h"]) for obj in horizontal) + max(0, len(horizontal) - 1)
    row_w = sum(int(obj["w"]) for obj in vertical) + max(0, len(vertical) - 1)
    bottom_h = max((int(obj["h"]) for obj in vertical), default=0)

    if horizontal and vertical:
        out_h = stack_h + 1 + bottom_h
        out_w = left_w + 1 + row_w
        row_start_col = left_w + 1
    elif horizontal:
        out_h = stack_h
        out_w = left_w
        row_start_col = 0
    else:
        out_h = bottom_h
        out_w = row_w
        row_start_col = 0

    out = np.full((out_h, out_w), BG_COLOR, dtype=int)

    row = 0
    for obj in horizontal:
        _paste_mask(out, obj["mask"], int(obj["color"]), row, 0)
        row += int(obj["h"]) + 1

    col = row_start_col
    for obj in vertical:
        top = out_h - int(obj["h"])
        _paste_mask(out, obj["mask"], int(obj["color"]), top, col)
        col += int(obj["w"]) + 1

    return out


def _random_mask(h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    if min(h, w) == 2 or random.random() < 0.35:
        return mask

    notch_h = random.randint(1, h - 1)
    notch_w = random.randint(1, w - 1)
    corner = random.choice(("tl", "tr", "bl", "br"))

    if corner == "tl":
        mask[:notch_h, :notch_w] = False
    elif corner == "tr":
        mask[:notch_h, w - notch_w :] = False
    elif corner == "bl":
        mask[h - notch_h :, :notch_w] = False
    else:
        mask[h - notch_h :, w - notch_w :] = False

    return mask


def _sample_object_specs() -> list[dict] | None:
    n_horizontal = random.randint(1, 3)
    n_vertical = random.randint(1, 3)
    total = n_horizontal + n_vertical
    colors = random.sample(FG_COLORS, total)

    specs: list[dict] = []
    used_h_areas: set[int] = set()
    used_v_areas: set[int] = set()
    color_idx = 0

    for orientation, count, used_areas in (
        ("h", n_horizontal, used_h_areas),
        ("v", n_vertical, used_v_areas),
    ):
        for _ in range(count):
            for _retry in range(80):
                if orientation == "h":
                    obj_h = random.randint(2, 5)
                    obj_w = random.randint(obj_h + 1, min(8, obj_h + 4))
                else:
                    obj_w = random.randint(2, 5)
                    obj_h = random.randint(obj_w + 1, min(8, obj_w + 4))

                mask = _random_mask(obj_h, obj_w)
                area = int(mask.sum())
                if area in used_areas:
                    continue
                used_areas.add(area)
                specs.append(
                    {
                        "color": colors[color_idx],
                        "h": obj_h,
                        "w": obj_w,
                        "mask": mask,
                        "area": area,
                    }
                )
                color_idx += 1
                break
            else:
                return None

    return specs


def _free_positions(occupied: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    H, W = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(H - h + 1):
        for c in range(W - w + 1):
            r0 = max(0, r - 1)
            r1 = min(H, r + h + 1)
            c0 = max(0, c - 1)
            c1 = min(W, c + w + 1)
            if not occupied[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _mark_occupied(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    H, W = occupied.shape
    r0 = max(0, top - 1)
    r1 = min(H, top + h + 1)
    c0 = max(0, left - 1)
    c1 = min(W, left + w + 1)
    occupied[r0:r1, c0:c1] = True


def _place_objects(specs: list[dict], H: int, W: int) -> np.ndarray | None:
    grid = np.full((H, W), BG_COLOR, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    order = sorted(specs, key=lambda spec: (-(spec["h"] * spec["w"]), -spec["area"]))
    for spec in order:
        positions = _free_positions(occupied, int(spec["h"]), int(spec["w"]))
        if not positions:
            return None
        top, left = random.choice(positions)
        _paste_mask(grid, spec["mask"], int(spec["color"]), top, left)
        _mark_occupied(occupied, top, left, int(spec["h"]), int(spec["w"]))

    return grid


def _spec_signature(specs: list[dict]) -> list[tuple[int, int, int, int]]:
    return sorted((int(spec["color"]), int(spec["area"]), int(spec["h"]), int(spec["w"])) for spec in specs)


def _solver_ready(grid: np.ndarray, specs: list[dict]) -> bool:
    try:
        components = _find_components(grid, bg=BG_COLOR)
        out = main(grid)
    except Exception:
        return False

    if len(components) != len(specs):
        return False

    comp_sig = sorted((int(obj["color"]), int(obj["area"]), int(obj["h"]), int(obj["w"])) for obj in components)
    if comp_sig != _spec_signature(specs):
        return False

    return isinstance(out, np.ndarray) and out.ndim == 2 and out.size > 0


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_object_specs()
    if specs is None:
        return None

    padded_area = sum((int(spec["h"]) + 2) * (int(spec["w"]) + 2) for spec in specs)
    base_side = max(15, int(np.ceil(np.sqrt(padded_area))) + 2)
    H = random.randint(base_side, 24)
    W = random.randint(base_side, 24)

    grid = _place_objects(specs, H, W)
    if grid is None:
        return None

    if not _solver_ready(grid, specs):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {MAX_GENERATION_ATTEMPTS} attempts")
