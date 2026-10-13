import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GENERATION_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            out.append({"coords": coords, "bbox": _bbox(coords), "area": int(coords.shape[0])})
    return out


def _erase_blue_columns(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for component in _components(out == BLUE):
        _r0, c0, _r1, _c1 = component["bbox"]
        coords = component["coords"]
        erase = coords[coords[:, 1] < c0 + 2]
        out[erase[:, 0], erase[:, 1]] = BG
    return out


def _fill_single_red_gaps(grid: np.ndarray) -> np.ndarray:
    base = _erase_blue_columns(grid)
    h, w = base.shape
    fill = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(1, w - 1):
            if base[r, c] == BG and base[r, c - 1] == RED and base[r, c + 1] == RED:
                fill[r, c] = True

    for r in range(1, h - 1):
        for c in range(w):
            if base[r, c] == BG and base[r - 1, c] == RED and base[r + 1, c] == RED:
                fill[r, c] = True

    out = base.copy()
    out[fill] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each blue object's two local leftmost columns, then repair only one-cell red line breaks.

    Input structure:
    - The background is black.
    - There are several blue 4-connected foreground objects whose bounding boxes are at least 3 columns wide.
    - Red horizontal or vertical line segments may run beside or through those blue objects, and some red pixels may overlay parts of the blue shapes.

    Rule:
    - Find every blue 4-connected component and compute its own bounding box.
    - Within each component, erase blue pixels that lie in the first two columns of that component's local bounding box; red pixels stay red.
    - After the erasure, scan red straight segments and fill any horizontal or vertical break that is exactly one background cell wide.
    - Do not repair wider breaks; leave all remaining blue and red pixels unchanged.

    Key insight:
    - The deletion is measured in each object's local coordinates, so the same two-column erasure creates different visible red gap widths depending on where a line first becomes red inside that object.
    """
    return _fill_single_red_gaps(np.asarray(grid, dtype=int))


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1 and comps[0]["area"] == int(mask.sum())


def _sample_blue_object(targeted: bool) -> tuple[np.ndarray, int]:
    dims = random.choice([(3, 3), (3, 4), (4, 3)])
    h, w = dims
    corridor = random.randint(1, h - 2) if targeted else random.randint(0, h - 1)
    for _ in range(60):
        mask = np.ones((h, w), dtype=bool)
        removable = [(r, c) for r in range(h) for c in range(w) if r != corridor]
        random.shuffle(removable)
        remove_count = random.randint(1, 3 if h * w == 9 else 4)
        for r, c in removable[:remove_count]:
            mask[r, c] = False
        if not 6 <= int(mask.sum()) <= 12:
            continue
        if not _mask_connected(mask):
            continue
        coords = np.argwhere(mask)
        if _bbox(coords) != (0, 0, h - 1, w - 1):
            continue
        return mask.astype(int), corridor
    raise RuntimeError("failed to sample blue object")


def _boxes_separated(boxes: list[tuple[int, int, int, int]], new_box: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = new_box
    for or0, oc0, or1, oc1 in boxes:
        if not (r1 + pad < or0 or or1 + pad < r0 or c1 + pad < oc0 or oc1 + pad < c0):
            return False
    return True


def _paint_object(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    for dr, dc in np.argwhere(mask > 0):
        grid[r0 + dr, c0 + dc] = BLUE


def _clear_path(grid: np.ndarray, row: int, c_start: int, c_end: int, own_box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = own_box
    for c in range(c_start, c_end + 1):
        if r0 <= row <= r1 and c0 <= c <= c1:
            continue
        if grid[row, c] != BG:
            return False
    return True


def _place_vertical_decoy(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for _ in range(40):
        length = random.randint(3, min(6, h - 2))
        r0 = random.randint(1, h - length - 1)
        c = random.randint(1, w - 2)
        ok = True
        for rr in range(r0 - 1, r0 + length + 1):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != BG:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        grid[r0 : r0 + length, c] = RED
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 18)
    w = random.randint(15, 18)
    grid = np.full((h, w), BG, dtype=int)

    object_count = random.randint(3, 5)
    kinds = ["wide", "narrow"]
    if object_count >= 4 and random.random() < 0.65:
        kinds.append("zero")
    while len(kinds) < object_count:
        kinds.append("plain")
    tail = kinds[2:]
    random.shuffle(tail)
    kinds = kinds[:2] + tail

    objects = []
    boxes = []
    for kind in kinds:
        targeted = kind in {"wide", "narrow", "zero"}
        mask, corridor = _sample_blue_object(targeted=targeted)
        oh, ow = mask.shape
        placed = False
        for _ in range(80):
            c_min = 4 if targeted else 1
            c_max = w - ow - (4 if targeted else 1)
            r_min = 1
            r_max = h - oh - 1
            if c_min > c_max or r_min > r_max:
                return None
            r0 = random.randint(r_min, r_max)
            c0 = random.randint(c_min, c_max)
            box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            if not _boxes_separated(boxes, box, pad=1):
                continue
            boxes.append(box)
            objects.append(
                {
                    "kind": kind,
                    "mask": mask,
                    "corridor": corridor,
                    "r0": r0,
                    "c0": c0,
                    "h": oh,
                    "w": ow,
                    "box": box,
                }
            )
            _paint_object(grid, mask, r0, c0)
            placed = True
            break
        if not placed:
            return None

    expected_fills = []
    wide_specs = []

    for obj in objects:
        if obj["kind"] not in {"wide", "narrow", "zero"}:
            continue
        gap = {"wide": 2, "narrow": 1, "zero": 0}[obj["kind"]]
        row = obj["r0"] + obj["corridor"]
        c0 = obj["c0"]
        c1 = c0 + obj["w"] - 1
        left_len = random.randint(3, 4)
        start = c0 - left_len
        if start < 0:
            return None
        inside_len = obj["w"] - gap
        right_piece_len = random.randint(max(3, inside_len), min(8, inside_len + 2))
        right_ext = right_piece_len - inside_len
        end = c1 + right_ext
        if end >= w:
            return None
        if not _clear_path(grid, row, start, c0 - 1, obj["box"]):
            return None
        if not _clear_path(grid, row, c1 + 1, end, obj["box"]):
            return None
        if not np.all(obj["mask"][obj["corridor"], :] == 1):
            return None

        grid[row, start:c0] = RED
        grid[row, c0 + gap : end + 1] = RED

        if obj["kind"] == "narrow":
            expected_fills.append((row, c0))
        elif obj["kind"] == "wide":
            wide_specs.append((row, c0))

    for _ in range(random.randint(0, 2)):
        _place_vertical_decoy(grid)

    blue_components = _components(grid == BLUE)
    if len(blue_components) != object_count:
        return None
    if any(not (6 <= comp["area"] <= 12) for comp in blue_components):
        return None
    if any(comp["bbox"][3] - comp["bbox"][1] + 1 < 3 for comp in blue_components):
        return None

    erased = _erase_blue_columns(grid)
    out = main(grid)

    actual_fills = {tuple(map(int, rc)) for rc in np.argwhere((erased == BG) & (out == RED))}
    if actual_fills != set(expected_fills):
        return None
    if not expected_fills:
        return None

    for row, c0 in expected_fills:
        if not (erased[row, c0] == BG and erased[row, c0 - 1] == RED and erased[row, c0 + 1] == RED):
            return None
        if out[row, c0] != RED:
            return None

    if not wide_specs:
        return None
    for row, c0 in wide_specs:
        if c0 + 2 >= w:
            return None
        if not (erased[row, c0 - 1] == RED and erased[row, c0] == BG and erased[row, c0 + 1] == BG and erased[row, c0 + 2] == RED):
            return None
        if out[row, c0] != BG or out[row, c0 + 1] != BG:
            return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate valid input for opus/13/010/4")
