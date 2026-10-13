from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
MAROON = 9
INPUT_PALETTE = (3, 4, 5, 6, 7, 8)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c

            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                r0 = min(r0, int(cr))
                r1 = max(r1, int(cr))
                c0 = min(c0, int(cc))
                c1 = max(c1, int(cc))
                for dr, dc in NEIGHBORS4:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            bbox_h = int(r1 - r0 + 1)
            bbox_w = int(c1 - c0 + 1)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "bbox_shape": (bbox_h, bbox_w),
                    "center": (int((r0 + r1) // 2), int((c0 + c1) // 2)),
                }
            )

    return components


def _is_slender(component: dict) -> bool:
    bbox_h, bbox_w = component["bbox_shape"]
    short = min(int(bbox_h), int(bbox_w))
    long = max(int(bbox_h), int(bbox_w))
    return long >= 3 * short


def _sort_key(component: dict) -> tuple[int, int]:
    center_r, center_c = component["center"]
    return int(center_c), int(center_r)


def _draw_l_path(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    if c0 <= c1:
        out[r0, c0 : c1 + 1] = int(color)
    else:
        out[r0, c1 : c0 + 1] = int(color)
    if r0 <= r1:
        out[r0 : r1 + 1, c1] = int(color)
    else:
        out[r1 : r0 + 1, c1] = int(color)


def _connect_chain(out: np.ndarray, components: list[dict], color: int) -> None:
    ordered = sorted(components, key=_sort_key)
    for first, second in zip(ordered, ordered[1:]):
        _draw_l_path(out, first["center"], second["center"], color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split objects by bbox aspect ratio, recolor compact ones, and connect class centers.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    components = _find_components(g)

    slender = [component for component in components if _is_slender(component)]
    compact = [component for component in components if not _is_slender(component)]

    for component in compact:
        for r, c in component["coords"]:
            out[int(r), int(c)] = MAROON

    _connect_chain(out, slender, RED)
    _connect_chain(out, compact, BLUE)
    return out


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _make_slender_mask() -> np.ndarray:
    orientation = random.choice(("h", "v"))
    thickness = random.choice((1, 1, 3))
    min_length = max(3, 3 * thickness)
    lengths = [length for length in (3, 5, 7, 9) if length >= min_length]
    if not lengths:
        thickness = 1
        lengths = [3, 5, 7, 9]
    length = random.choice(lengths)
    if orientation == "h":
        return np.ones((thickness, length), dtype=bool)
    return np.ones((length, thickness), dtype=bool)


def _make_compact_mask() -> np.ndarray:
    dims = [
        (h, w)
        for h in (3, 5, 7)
        for w in (3, 5, 7)
        if max(h, w) < 3 * min(h, w)
    ]
    h, w = random.choice(dims)
    mask = np.ones((h, w), dtype=bool)
    center = (h // 2, w // 2)
    mandatory = {
        center,
        (0, w // 2),
        (h - 1, w // 2),
        (h // 2, 0),
        (h // 2, w - 1),
    }

    min_area = max(5, int(np.ceil(0.6 * h * w)))
    target_area = random.randint(min_area, h * w)
    removable = [(r, c) for r in range(h) for c in range(w) if (r, c) not in mandatory]
    random.shuffle(removable)

    for r, c in removable:
        if int(mask.sum()) <= target_area:
            break
        mask[r, c] = False
        if not _mask_is_connected(mask) or _bbox_from_mask(mask) != (0, 0, h - 1, w - 1):
            mask[r, c] = True

    return mask


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


def _path_cells(start: tuple[int, int], end: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    cells = {(int(r0), int(c)) for c in range(min(c0, c1), max(c0, c1) + 1)}
    cells.update((int(r), int(c1)) for r in range(min(r0, r1), max(r0, r1) + 1))
    return cells


def _place_objects(height: int, width: int, objects: list[dict]) -> bool:
    occupied = np.zeros((height, width), dtype=bool)
    forbidden = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(objects)), key=lambda idx: int(objects[idx]["mask"].sum()), reverse=True)

    for idx in order:
        obj = objects[idx]
        mask = obj["mask"]
        h, w = mask.shape
        row_range = range(1, height - h) if height - h >= 2 else range(0, height - h + 1)
        col_range = range(1, width - w) if width - w >= 2 else range(0, width - w + 1)
        positions = [(r, c) for r in row_range for c in col_range]
        random.shuffle(positions)

        placed = False
        for top, left in positions:
            global_mask = np.zeros((height, width), dtype=bool)
            global_mask[top : top + h, left : left + w] = mask
            if bool((global_mask & forbidden).any()):
                continue
            occupied |= global_mask
            forbidden = _dilate8(occupied)
            obj["top"] = int(top)
            obj["left"] = int(left)
            obj["cells"] = {(int(top + r), int(left + c)) for r, c in np.argwhere(mask)}
            obj["center"] = (int(top + h // 2), int(left + w // 2))
            placed = True
            break

        if not placed:
            return False

    return True


def _paths_are_clear(objects: list[dict]) -> bool:
    owner: dict[tuple[int, int], int] = {}
    for idx, obj in enumerate(objects):
        for cell in obj["cells"]:
            owner[cell] = idx

    red_cells: set[tuple[int, int]] = set()
    blue_cells: set[tuple[int, int]] = set()

    for kind, path_cells in (("slender", red_cells), ("compact", blue_cells)):
        group = [obj for obj in objects if obj["kind"] == kind]
        ordered = sorted(group, key=_sort_key)
        for first, second in zip(ordered, ordered[1:]):
            segment = _path_cells(first["center"], second["center"])
            allowed = {int(first["id"]), int(second["id"])}
            for cell in segment:
                if cell in owner and int(owner[cell]) not in allowed:
                    return False
            path_cells.update(segment)

    return not bool(red_cells & blue_cells)


def _render_input(height: int, width: int, objects: list[dict]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for obj in objects:
        for r, c in obj["cells"]:
            grid[int(r), int(c)] = int(obj["color"])
    return grid


def _sample_objects() -> list[dict]:
    slender_count, compact_count = random.choice(((2, 2), (2, 3), (3, 2), (3, 3)))
    total = slender_count + compact_count
    colors = random.sample(INPUT_PALETTE, total)
    objects: list[dict] = []

    for idx in range(slender_count):
        objects.append(
            {
                "id": len(objects),
                "kind": "slender",
                "color": colors[idx],
                "mask": _make_slender_mask(),
            }
        )

    for idx in range(compact_count):
        objects.append(
            {
                "id": len(objects),
                "kind": "compact",
                "color": colors[slender_count + idx],
                "mask": _make_compact_mask(),
            }
        )

    random.shuffle(objects)
    for idx, obj in enumerate(objects):
        obj["id"] = idx
    return objects


def _is_valid_example(inp: np.ndarray, objects: list[dict]) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False

    if not np.any(out == RED):
        return False
    if not np.any(out == BLUE):
        return False
    if not np.any(out == MAROON):
        return False

    components = _find_components(inp)
    if len(components) != len(objects):
        return False
    if sum(_is_slender(component) for component in components) < 2:
        return False
    if sum(not _is_slender(component) for component in components) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(18, 30)
    objects = _sample_objects()
    if not _place_objects(height, width, objects):
        return None
    if not _paths_are_clear(objects):
        return None
    inp = _render_input(height, width, objects)
    if _is_valid_example(inp, objects):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
