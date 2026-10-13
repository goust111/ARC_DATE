from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    try:
        return SourceFileLoader("_arc_utils_for_025_4", str(path)).load_module()
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


if _ARC_UTILS is not None:
    Color = _ARC_UTILS.Color
    bbox_coords = _ARC_UTILS.bbox_coords
    crop_bbox = _ARC_UTILS.crop_bbox
    components_mask = _ARC_UTILS.components_mask
else:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        if connectivity != 4:
            raise ValueError("fallback components_mask only supports 4-connectivity")
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        comps: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(coords, dtype=int))
        return comps


BG = Color.BLACK
DIVIDER = Color.GREY
DIFF = Color.RED
COLOR_POOL = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON)
MAX_GENERATION_ATTEMPTS = 400


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _rows_from_mask(mask: np.ndarray) -> list[str]:
    return ["".join("1" if cell else "0" for cell in row) for row in mask.astype(int)]


def _transposed_rows(rows: list[str]) -> list[str]:
    mask = _mask(rows)
    return _rows_from_mask(mask.T)


_BASE_SHAPES = {
    "sym": {
        (2, 2): [["11", "11"]],
        (2, 3): [["110", "011"]],
        (2, 4): [["1110", "0111"]],
        (3, 3): [["010", "111", "010"], ["110", "111", "011"]],
        (3, 4): [["1100", "1111", "0011"], ["1000", "1111", "0001"]],
    },
    "asym": {
        (2, 2): [["11", "10"]],
        (2, 3): [["111", "100"], ["111", "101"]],
        (2, 4): [["1111", "1000"], ["1111", "1010"]],
        (3, 3): [["110", "100", "111"], ["111", "101", "100"]],
        (3, 4): [["1111", "1001", "1000"], ["1110", "1000", "1111"]],
    },
}


def _build_shape_library() -> dict[str, dict[tuple[int, int], list[np.ndarray]]]:
    raw: dict[str, dict[tuple[int, int], list[list[str]]]] = {"sym": {}, "asym": {}}
    for kind, shapes_by_size in _BASE_SHAPES.items():
        for size, shape_rows_list in shapes_by_size.items():
            raw[kind].setdefault(size, []).extend(shape_rows_list)
            h, w = size
            if h != w:
                t_size = (w, h)
                raw[kind].setdefault(t_size, [])
                for rows in shape_rows_list:
                    raw[kind][t_size].append(_transposed_rows(rows))

    library: dict[str, dict[tuple[int, int], list[np.ndarray]]] = {"sym": {}, "asym": {}}
    for kind, shapes_by_size in raw.items():
        for size, shape_rows_list in shapes_by_size.items():
            seen: set[tuple[bytes, tuple[int, int]]] = set()
            variants: list[np.ndarray] = []
            for rows in shape_rows_list:
                base = _mask(rows)
                transforms = [base, np.flipud(base), np.fliplr(base), np.rot90(base, 2)]
                if size[0] == size[1]:
                    transforms.extend([np.rot90(base, 1), np.rot90(base, 3)])
                for mask in transforms:
                    key = (mask.tobytes(), mask.shape)
                    if key in seen:
                        continue
                    seen.add(key)
                    assert 3 <= int(mask.sum()) <= 8
                    variants.append(mask.copy())
            library[kind][size] = variants
    return library


SHAPE_LIBRARY = _build_shape_library()
SHAPE_SIZES = tuple(sorted(SHAPE_LIBRARY["sym"].keys()))


def _find_divider_col(grid: np.ndarray) -> int:
    cols = np.where(np.all(grid == DIVIDER, axis=0))[0]
    assert len(cols) == 1, "expected exactly one full-height grey divider column"
    divider_col = int(cols[0])
    left_w = divider_col
    right_w = int(grid.shape[1] - divider_col - 1)
    assert left_w == right_w and left_w > 0, "divider must split the grid into equal-width panels"
    return divider_col


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    divider_col = _find_divider_col(grid)
    return grid[:, :divider_col].copy(), grid[:, divider_col + 1 :].copy()


def _panel_components(panel: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(c) for c in np.unique(panel) if int(c) != BG):
        for coords in components_mask(panel == color, connectivity=4):
            comp_bbox = bbox_coords(coords)
            patch = crop_bbox(panel, comp_bbox)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": comp_bbox,
                    "mask": patch == color,
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _rotate_panel_locally(panel: np.ndarray) -> np.ndarray:
    rotated = np.zeros_like(panel)
    for comp in _panel_components(panel):
        r0, c0, r1, c1 = comp["bbox"]
        color = int(comp["color"])
        local = np.rot90(comp["mask"], 2)
        view = rotated[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(view[local] != BG), "rotated object bboxes must stay disjoint"
        view[local] = color
    return rotated


def _diff_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    out = np.zeros(left.shape, dtype=int)
    out[left != right] = DIFF
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the left panel with the right panel after rotating each right-hand object 180 degrees inside its own bbox.

    Input structure:
    - A black background is split by one full-height grey divider column into equal-width left and right panels.
    - Each panel contains 2-4 separated monochromatic 4-connected objects using non-black, non-grey colors.
    - Matching-color objects occupy the same bbox positions relative to their own panel origins.

    Rule:
    - Find the grey divider and split the grid into the two panels.
    - In the right panel, rotate every connected foreground object by 180 degrees inside its own bbox, keeping the same color.
    - Compare the original left panel and the locally rotated right panel cell by cell; output black where the cells are identical and red where they differ.

    Key insight:
    - The decisive comparison is not left-versus-right as drawn, but left-versus-right after each right-hand object is independently half-turned inside its own local frame.
    """
    g = np.asarray(grid, dtype=int)
    left, right = _split_panels(g)
    rotated_right = _rotate_panel_locally(right)
    return _diff_mask(left, rotated_right)


def _choose_shape(kind: str, max_h: int, max_w: int) -> np.ndarray:
    sizes = [size for size in SHAPE_SIZES if size[0] <= max_h and size[1] <= max_w]
    assert sizes, "panel is too small for any object bbox"
    size = random.choice(sizes)
    return random.choice(SHAPE_LIBRARY[kind][size]).copy()


def _place_objects(panel_h: int, panel_w: int, masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(masks)
    blocked = np.zeros((panel_h, panel_w), dtype=bool)
    order = sorted(range(len(masks)), key=lambda i: (-masks[i].shape[0] * masks[i].shape[1], -int(masks[i].sum())))

    def search(depth: int) -> bool:
        if depth == len(order):
            return True
        idx = order[depth]
        h, w = masks[idx].shape
        candidates = [
            (r, c)
            for r in range(1, panel_h - h)
            for c in range(1, panel_w - w)
            if not blocked[max(0, r - 1) : min(panel_h, r + h + 1), max(0, c - 1) : min(panel_w, c + w + 1)].any()
        ]
        random.shuffle(candidates)
        for r, c in candidates:
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(panel_h, r + h + 1)
            cc1 = min(panel_w, c + w + 1)
            blocked[rr0:rr1, cc0:cc1] = True
            positions[idx] = (r, c)
            if search(depth + 1):
                return True
            positions[idx] = None
            blocked[rr0:rr1, cc0:cc1] = False
        return False

    if not search(0):
        return None
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _paint_panel(panel_shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    panel = np.zeros(panel_shape, dtype=int)
    for obj in objects:
        r0, c0 = obj["top_left"]
        mask = obj["mask"]
        color = int(obj["color"])
        view = panel[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        assert not np.any(view[mask] != BG), "objects must not overlap"
        view[mask] = color
    return panel


def _expand_bbox(top: int, left: int, mask: np.ndarray) -> tuple[int, int, int, int]:
    return int(top), int(left), int(top + mask.shape[0] - 1), int(left + mask.shape[1] - 1)


def _object_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _build_case() -> tuple[np.ndarray, dict] | None:
    n_objects = random.randint(2, 4)
    min_side = max(10, 2 * n_objects + 3)
    panel_h = random.randint(min_side, 14)
    panel_w = random.randint(min_side, 14)

    kinds = ["sym", "asym"] + [random.choice(["sym", "asym"]) for _ in range(n_objects - 2)]
    random.shuffle(kinds)
    colors = random.sample(COLOR_POOL, n_objects)
    masks = [_choose_shape(kind, max_h=4, max_w=4) for kind in kinds]
    positions = _place_objects(panel_h, panel_w, masks)
    if positions is None:
        return None

    objects = []
    for color, kind, mask, top_left in zip(colors, kinds, masks, positions):
        r0, c0 = top_left
        obj = {
            "color": int(color),
            "kind": kind,
            "mask": mask.copy(),
            "top_left": (int(r0), int(c0)),
            "bbox": _expand_bbox(r0, c0, mask),
        }
        objects.append(obj)

    left = _paint_panel((panel_h, panel_w), objects)
    right = _paint_panel((panel_h, panel_w), objects)
    expected = _diff_mask(left, _rotate_panel_locally(right))

    if not np.any(expected == DIFF):
        return None
    if np.any(_diff_mask(left, right)):
        return None
    if np.array_equal(expected, _diff_mask(left, np.rot90(right, 2))):
        return None

    grid = np.full((panel_h, panel_w * 2 + 1), BG, dtype=int)
    grid[:, :panel_w] = left
    grid[:, panel_w] = DIVIDER
    grid[:, panel_w + 1 :] = right

    metadata = {
        "panel_shape": (panel_h, panel_w),
        "objects": objects,
        "left": left,
        "right": right,
        "expected": expected,
    }
    return grid, metadata


def _validate_metadata(grid: np.ndarray, metadata: dict) -> bool:
    left, right = _split_panels(grid)
    if not np.array_equal(left, metadata["left"]):
        return False
    if not np.array_equal(right, metadata["right"]):
        return False

    expected = metadata["expected"]
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    divider_col = _find_divider_col(grid)
    if divider_col != metadata["panel_shape"][1]:
        return False

    objects = metadata["objects"]
    if len(objects) < 2 or len(objects) > 4:
        return False

    left_comps = _panel_components(left)
    right_comps = _panel_components(right)
    if len(left_comps) != len(objects) or len(right_comps) != len(objects):
        return False

    seen_bboxes: set[tuple[int, int, int, int]] = set()
    sym_count = 0
    asym_count = 0
    for obj in objects:
        mask = obj["mask"]
        bbox = obj["bbox"]
        color = int(obj["color"])
        if bbox in seen_bboxes:
            return False
        seen_bboxes.add(bbox)

        r0, c0, r1, c1 = bbox
        if not (3 <= int(mask.sum()) <= 8):
            return False
        if np.any(left[r0 : r1 + 1, c0 : c1 + 1][mask] != color):
            return False
        if np.any(right[r0 : r1 + 1, c0 : c1 + 1][mask] != color):
            return False

        rotated_mask = np.rot90(mask, 2)
        box_out = expected[r0 : r1 + 1, c0 : c1 + 1]
        if obj["kind"] == "sym":
            sym_count += 1
            if not np.array_equal(rotated_mask, mask):
                return False
            if np.any(box_out != BG):
                return False
        else:
            asym_count += 1
            if np.array_equal(rotated_mask, mask):
                return False
            if not np.any(box_out == DIFF):
                return False
            local_expected = np.zeros(mask.shape, dtype=int)
            local_expected[mask != rotated_mask] = DIFF
            if not np.array_equal(box_out, local_expected):
                return False

    if sym_count == 0 or asym_count == 0:
        return False

    no_rotation = _diff_mask(left, right)
    if np.any(no_rotation):
        return False
    if not np.any(expected):
        return False
    if np.array_equal(expected, _diff_mask(left, np.rot90(right, 2))):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, metadata = built
        if _validate_metadata(grid, metadata):
            return grid
    raise RuntimeError("failed to generate a valid local-rotation XOR puzzle")
