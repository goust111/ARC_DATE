from __future__ import annotations

from functools import lru_cache
import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PINK = 6
ORANGE = 7

TYPE_ORDER = ("A", "B", "C", "D")
TYPE_TO_ROW = {"A": 0, "B": 1, "C": 2, "D": 3}
ROW_TO_EXPECTED_COLOR = (RED, GREEN, BLUE, ORANGE)

GRID_H_RANGE = (16, 23)
GRID_W_RANGE = (18, 24)
PANEL_SHAPE = (4, 5)
MAX_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_coords(coords)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _classify_mask(mask: np.ndarray) -> str:
    shape = _normalize_mask(mask)
    has_h_reflection = np.array_equal(shape, np.flipud(shape))
    has_v_reflection = np.array_equal(shape, np.fliplr(shape))
    has_rot180 = np.array_equal(shape, np.rot90(shape, 2))
    reflection_count = int(has_h_reflection) + int(has_v_reflection)
    if reflection_count == 0 and not has_rot180:
        return "A"
    if reflection_count == 1:
        return "B"
    if reflection_count == 2:
        return "C"
    if reflection_count == 0 and has_rot180:
        return "D"
    raise AssertionError("shape does not fit one of the four intended symmetry classes")


def _legend_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = _bbox(grid == GREY)
    assert box is not None, "expected a grey legend panel"
    r0, c0, r1, c1 = box
    assert (r1 - r0 + 1, c1 - c0 + 1) == PANEL_SHAPE, "legend panel must be 4x5"
    return box


def _legend_mapping(grid: np.ndarray) -> dict[str, int]:
    r0, c0, r1, c1 = _legend_box(grid)
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = []
    for row in range(PANEL_SHAPE[0]):
        assert panel[row, 0] == GREY and panel[row, -1] == GREY
        bar = panel[row, 1:4]
        assert len(set(map(int, bar))) == 1, "each legend row must contain one solid 1x3 bar"
        color = int(bar[0])
        assert color != GREY
        colors.append(color)
    return {typ: colors[idx] for idx, typ in enumerate(TYPE_ORDER)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each magenta object with the legend color indexed by its symmetry type.

    Input structure
    - A black background contains a 4x5 grey legend panel on the left.
    - Each legend row has a centered 1x3 color bar, from top to bottom encoding the four symmetry types.
    - To the right are 3-5 disconnected magenta 4-connected objects, each occupying at most a 3x3 bbox.

    Rule
    - Classify every magenta object inside its own tight bbox as A: no reflection and no 180-degree rotation, B: exactly one reflection axis, C: two orthogonal reflection axes, or D: 180-degree rotation without reflection.
    - Use that type as a lookup key into the legend rows.
    - Fill the object's entire bbox with the looked-up color, then place the original magenta pixels back on top.
    - Leave the legend panel and all other background pixels unchanged.

    Key insight
    - The legend is a type-to-color dictionary, and the required type is determined only in the object's local bbox coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC grid"

    lookup = _legend_mapping(g)
    components = _components(g == PINK)
    assert 3 <= len(components) <= 5, "expected 3-5 magenta objects"

    out = g.copy()
    for coords in components:
        r0, c0, r1, c1 = _bbox_coords(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        sym_type = _classify_mask(local)
        fill_color = lookup[sym_type]
        out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
        out[coords[:, 0], coords[:, 1]] = PINK
    return out


@lru_cache(maxsize=1)
def _shape_bank() -> dict[str, tuple[np.ndarray, ...]]:
    by_type: dict[str, list[np.ndarray]] = {typ: [] for typ in TYPE_ORDER}
    seen: dict[str, set[tuple[tuple[int, int], bytes]]] = {typ: set() for typ in TYPE_ORDER}
    for height in range(1, 4):
        for width in range(1, 4):
            area = height * width
            for bits in range(1, 1 << area):
                mask = np.zeros((height, width), dtype=bool)
                for idx in range(area):
                    if (bits >> idx) & 1:
                        mask[idx // width, idx % width] = True
                if not (3 <= int(mask.sum()) <= 6):
                    continue
                if len(_components(mask)) != 1:
                    continue
                norm = _normalize_mask(mask)
                if np.all(norm):
                    continue
                sym_type = _classify_mask(norm)
                key = (tuple(map(int, norm.shape)), norm.tobytes())
                if key in seen[sym_type]:
                    continue
                seen[sym_type].add(key)
                by_type[sym_type].append(norm.copy())
    return {typ: tuple(by_type[typ]) for typ in TYPE_ORDER}


def _pick_shape(
    rng: np.random.Generator,
    sym_type: str,
    *,
    dims: tuple[int, int] | None = None,
    area: int | None = None,
) -> np.ndarray:
    candidates = []
    for shape in _shape_bank()[sym_type]:
        if dims is not None and tuple(map(int, shape.shape)) != tuple(map(int, dims)):
            continue
        if area is not None and int(shape.sum()) != int(area):
            continue
        candidates.append(shape)
    assert candidates, f"no candidate shape for {sym_type} with dims={dims}, area={area}"
    return candidates[int(rng.integers(len(candidates)))].copy()


def _selected_types(rng: np.random.Generator) -> list[str]:
    count = int(rng.choice([3, 4, 5], p=[0.25, 0.45, 0.30]))
    if count == 3:
        missing = TYPE_ORDER[int(rng.integers(len(TYPE_ORDER)))]
        return [typ for typ in TYPE_ORDER if typ != missing]
    if count == 4:
        return list(TYPE_ORDER)
    extra = TYPE_ORDER[int(rng.integers(len(TYPE_ORDER)))]
    return list(TYPE_ORDER) + [extra]


def _choose_shapes(rng: np.random.Generator, types: list[str]) -> list[tuple[str, np.ndarray]]:
    family = (2, 3) if int(rng.integers(2)) == 0 else (3, 2)
    chosen: list[tuple[str, np.ndarray]] = []
    for sym_type in types:
        if sym_type == "C":
            shape = _pick_shape(rng, "C")
        elif len(chosen) < len(TYPE_ORDER):
            shape = _pick_shape(rng, sym_type, dims=family, area=4)
        else:
            if rng.random() < 0.5:
                shape = _pick_shape(rng, sym_type, dims=family)
            else:
                shape = _pick_shape(rng, sym_type)
        chosen.append((sym_type, shape))
    rng.shuffle(chosen)
    return chosen


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(patch)
    grid[r0 + coords[:, 0], c0 + coords[:, 1]] = color


def _mark_forbidden(mask: np.ndarray, box: tuple[int, int, int, int], pad: int = 1) -> None:
    height, width = mask.shape
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(height - 1, r1 + pad)
    cc1 = min(width - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_objects(
    rng: np.random.Generator,
    grid: np.ndarray,
    shapes: list[tuple[str, np.ndarray]],
    panel_box: tuple[int, int, int, int],
) -> list[dict] | None:
    height, width = grid.shape
    forbidden = np.zeros((height, width), dtype=bool)
    _mark_forbidden(forbidden, panel_box, pad=1)
    work_c0 = panel_box[3] + 2
    placed: list[dict | None] = [None] * len(shapes)

    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][1].shape[0] * shapes[idx][1].shape[1], reverse=True)
    for idx in order:
        sym_type, shape = shapes[idx]
        sh, sw = map(int, shape.shape)
        candidates = []
        for r0 in range(0, height - sh + 1):
            for c0 in range(work_c0, width - sw + 1):
                if forbidden[r0 : r0 + sh, c0 : c0 + sw].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        _paint_patch(grid, shape, r0, c0, PINK)
        box = (r0, c0, r0 + sh - 1, c0 + sw - 1)
        _mark_forbidden(forbidden, box, pad=1)
        placed[idx] = {"type": sym_type, "shape": shape.copy(), "bbox": box}

    return [item for item in placed if item is not None]


def _build_panel(grid: np.ndarray, r0: int, c0: int) -> tuple[int, int, int, int]:
    panel = np.full(PANEL_SHAPE, GREY, dtype=int)
    for row, color in enumerate(ROW_TO_EXPECTED_COLOR):
        panel[row, 1:4] = color
    grid[r0 : r0 + PANEL_SHAPE[0], c0 : c0 + PANEL_SHAPE[1]] = panel
    return (r0, c0, r0 + PANEL_SHAPE[0] - 1, c0 + PANEL_SHAPE[1] - 1)


def _object_summaries(grid: np.ndarray) -> list[dict]:
    summaries = []
    for coords in _components(grid == PINK):
        r0, c0, r1, c1 = _bbox_coords(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        summaries.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": local,
                "type": _classify_mask(local),
                "area": int(local.sum()),
                "dims": tuple(map(int, local.shape)),
            }
        )
    return summaries


def _validate_input(grid: np.ndarray) -> bool:
    try:
        box = _legend_box(grid)
        mapping = _legend_mapping(grid)
        if tuple(mapping[typ] for typ in TYPE_ORDER) != ROW_TO_EXPECTED_COLOR:
            return False

        r0, c0, r1, c1 = box
        if c0 > 2:
            return False

        objs = _object_summaries(grid)
        if not (3 <= len(objs) <= 5):
            return False
        types = [obj["type"] for obj in objs]
        if len(objs) == 3:
            if len(set(types)) != 3:
                return False
        elif set(types) != set(TYPE_ORDER):
            return False

        for obj in objs:
            if not (3 <= obj["area"] <= 6):
                return False
            if max(obj["dims"]) > 3:
                return False
            if np.all(obj["mask"]):
                return False
            br0, bc0, br1, bc1 = obj["bbox"]
            if br0 <= r1 + 1 and bc0 <= c1 + 1:
                return False

        same_dims_cross_type = False
        same_area_cross_type = False
        for i, a in enumerate(objs):
            for b in objs[i + 1 :]:
                if a["type"] == b["type"]:
                    continue
                same_dims_cross_type |= a["dims"] == b["dims"]
                same_area_cross_type |= a["area"] == b["area"]
        if not same_dims_cross_type or not same_area_cross_type:
            return False

        out = main(grid)
        for obj in objs:
            br0, bc0, br1, bc1 = obj["bbox"]
            fill_color = mapping[obj["type"]]
            bbox_view = out[br0 : br1 + 1, bc0 : bc1 + 1]
            if not np.all((bbox_view == fill_color) | (bbox_view == PINK)):
                return False
            if not np.any(bbox_view == fill_color):
                return False
        return True
    except Exception:
        return False


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(*GRID_H_RANGE))
    width = int(rng.integers(*GRID_W_RANGE))
    grid = np.full((height, width), BG, dtype=int)

    panel_r0 = int(rng.integers(0, height - PANEL_SHAPE[0] + 1))
    panel_c0 = int(rng.integers(0, 3))
    panel_box = _build_panel(grid, panel_r0, panel_c0)

    shapes = _choose_shapes(rng, _selected_types(rng))
    placed = _place_objects(rng, grid, shapes, panel_box)
    if placed is None:
        return None

    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-orbit lookup-stamp input")
