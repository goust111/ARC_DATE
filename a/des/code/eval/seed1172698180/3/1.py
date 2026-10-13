from __future__ import annotations

import random
import numpy as np


BLACK = 0
RED = 2
GREY = 5
TEMPLATE_COLORS = (1, 3, 4, 6, 7, 8, 9)

ORIENTATIONS = ("up", "right", "down", "left")
ROTATION_STEPS_CCW = {"up": 0, "right": 3, "down": 2, "left": 1}
L_SHAPES = {
    "up": {(0, 0), (0, 1), (1, 0)},
    "right": {(0, 0), (0, 1), (1, 1)},
    "down": {(0, 1), (1, 0), (1, 1)},
    "left": {(0, 0), (1, 0), (1, 1)},
}

FRAME_SIZE_CHOICES = (7, 9, 11)
GRID_HEIGHT_RANGE = (20, 33)
GRID_WIDTH_RANGE = (20, 33)
ORIENTATION_KIND_RANGE = (2, 4)
EXTRA_FRAME_RANGE = (0, 2)
OBJECT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400

_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _reading_key(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return int(bbox[0]), int(bbox[1])


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    base = np.asarray(mask, dtype=bool)
    for k in range(4):
        rot = np.rot90(base, k)
        variants.append(rot)
        variants.append(np.fliplr(rot))
    return variants


def _canonical_mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = _dihedral_variants(mask)
    return min((tuple(map(int, v.shape)), v.tobytes()) for v in variants)


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _is_asymmetric(mask: np.ndarray) -> bool:
    return sum(np.array_equal(mask, variant) for variant in _dihedral_variants(mask)) == 1


def _build_template_library() -> tuple[np.ndarray, ...]:
    reps: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for bits in range(1 << 9):
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if (bits >> idx) & 1:
                mask[idx // 3, idx % 3] = True
        area = int(mask.sum())
        if area < 4 or area > 6:
            continue
        if not _is_connected(mask):
            continue
        if not _is_asymmetric(mask):
            continue
        key = _canonical_mask_key(mask)
        reps.setdefault(key, mask.copy())
    library = tuple(reps[key] for key in sorted(reps))
    assert len(library) >= 4
    return library


TEMPLATE_MASK_LIBRARY = _build_template_library()


def _orientation_from_l(coords: list[tuple[int, int]]) -> str:
    assert len(coords) == 3
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 == 1 and c1 - c0 == 1, "red L must occupy a 2x2 box"
    rel = {(r - r0, c - c0) for r, c in coords}
    for orientation, shape in L_SHAPES.items():
        if rel == shape:
            return orientation
    raise AssertionError(f"unrecognized L shape: {sorted(rel)}")


def _extract_frames(g: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        side = r1 - r0 + 1
        if side < 5 or c1 - c0 + 1 != side:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        if set(comp) != border:
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        red_coords = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(g[r, c]) == RED
        ]
        assert len(red_coords) == 3, "each frame must contain exactly one red L"
        assert np.all((inner == BLACK) | (inner == RED)), "frame interior must only contain black/red"
        orientation = _orientation_from_l(red_coords)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "orientation": orientation,
                "red_coords": red_coords,
                "inner_size": side - 2,
            }
        )
    frames.sort(key=lambda frame: _reading_key(frame["bbox"]))  # type: ignore[index]
    assert frames, "no valid grey frames found"
    return frames


def _extract_templates(g: np.ndarray) -> list[dict[str, object]]:
    templates: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in {BLACK, RED, GREY}):
        comps = _components(g == color)
        assert len(comps) == 1, f"template color {color} must form exactly one component"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
        for r, c in coords:
            patch[r - r0, c - c0] = color
        templates.append({"color": color, "bbox": (r0, c0, r1, c1), "patch": patch})
    templates.sort(key=lambda item: _reading_key(item["bbox"]))  # type: ignore[index]
    assert templates, "no external templates found"
    return templates


def _paste_with_transparency(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = out[top : top + h, left : left + w]
    assert target.shape == patch.shape
    mask = patch != BLACK
    assert not np.any((target != BLACK) & mask), "template overlaps non-empty target"
    target[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey frame with the template selected by its red L orientation.

    The red 3-cell L is read from the missing corner of its 2x2 bbox. The four
    resulting states are named `up/right/down/left` in rotation order, with the
    canonical `up` shape missing its lower-right corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_frames(g)
    templates = _extract_templates(g)

    orientation_order: list[str] = []
    for frame in frames:
        orientation = str(frame["orientation"])
        if orientation not in orientation_order:
            orientation_order.append(orientation)
    assert len(orientation_order) == len(templates), "template count must equal used orientation kinds"
    template_for_orientation = {
        orientation: np.asarray(templates[idx]["patch"], dtype=int)
        for idx, orientation in enumerate(orientation_order)
    }

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        inner_size = int(frame["inner_size"])
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[inner == RED] = BLACK

        orientation = str(frame["orientation"])
        base_patch = template_for_orientation[orientation]
        patch = np.rot90(base_patch, ROTATION_STEPS_CCW[orientation])
        ph, pw = map(int, patch.shape)
        assert ph < inner_size and pw < inner_size, "template bbox must be strictly smaller than frame interior"
        top = r0 + 1 + (inner_size - ph) // 2
        left = c0 + 1 + (inner_size - pw) // 2
        _paste_with_transparency(out, patch, top, left)

    return out


def _free_positions(occupied: np.ndarray, h: int, w: int, pad: int = OBJECT_PADDING) -> list[tuple[int, int]]:
    H, W = occupied.shape
    positions: list[tuple[int, int]] = []
    for r in range(1, H - h):
        for c in range(1, W - w):
            rr0 = max(0, r - pad)
            cc0 = max(0, c - pad)
            rr1 = min(H, r + h + pad)
            cc1 = min(W, c + w + pad)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r, c))
    return positions


def _reserve(occupied: np.ndarray, r: int, c: int, h: int, w: int, pad: int = OBJECT_PADDING) -> None:
    H, W = occupied.shape
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(H, r + h + pad)
    cc1 = min(W, c + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    grid[r, c : c + size] = GREY
    grid[r + size - 1, c : c + size] = GREY
    grid[r : r + size, c] = GREY
    grid[r : r + size, c + size - 1] = GREY


def _draw_l(grid: np.ndarray, top: int, left: int, orientation: str) -> None:
    for dr, dc in L_SHAPES[orientation]:
        grid[top + dr, left + dc] = RED


def _random_template_masks(count: int) -> list[np.ndarray]:
    choices = random.sample(range(len(TEMPLATE_MASK_LIBRARY)), count)
    return [TEMPLATE_MASK_LIBRARY[idx].copy() for idx in choices]


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_HEIGHT_RANGE)
    W = random.randint(*GRID_WIDTH_RANGE)
    kind_count = random.randint(*ORIENTATION_KIND_RANGE)
    extra_frames = random.randint(*EXTRA_FRAME_RANGE)
    frame_count = min(kind_count + extra_frames, 5)

    frame_sizes = [random.choice(FRAME_SIZE_CHOICES) for _ in range(frame_count)]
    frame_indices = sorted(range(frame_count), key=lambda idx: frame_sizes[idx], reverse=True)
    occupied = np.zeros((H, W), dtype=bool)
    frame_specs: list[dict[str, int]] = [{} for _ in range(frame_count)]  # type: ignore[list-item]
    for idx in frame_indices:
        size = frame_sizes[idx]
        positions = _free_positions(occupied, size, size)
        if not positions:
            return None
        r, c = random.choice(positions)
        _reserve(occupied, r, c, size, size)
        frame_specs[idx] = {"r": r, "c": c, "size": size}

    frame_specs.sort(key=lambda spec: (int(spec["r"]), int(spec["c"])))
    orientation_order = random.sample(list(ORIENTATIONS), kind_count)
    frame_orientations = orientation_order + [random.choice(orientation_order) for _ in range(frame_count - kind_count)]
    for spec, orientation in zip(frame_specs, frame_orientations):
        spec["orientation"] = orientation

    template_positions: list[tuple[int, int]] = []
    for _ in range(kind_count):
        positions = _free_positions(occupied, 3, 3)
        if not positions:
            return None
        r, c = random.choice(positions)
        _reserve(occupied, r, c, 3, 3)
        template_positions.append((r, c))
    template_positions.sort()

    template_masks = _random_template_masks(kind_count)
    template_colors = random.sample(TEMPLATE_COLORS, kind_count)

    grid = np.full((H, W), BLACK, dtype=int)
    for spec in frame_specs:
        r = int(spec["r"])
        c = int(spec["c"])
        size = int(spec["size"])
        _draw_frame(grid, r, c, size)
        l_top = random.randint(r + 1, r + size - 3)
        l_left = random.randint(c + 1, c + size - 3)
        _draw_l(grid, l_top, l_left, str(spec["orientation"]))

    for (r, c), mask, color in zip(template_positions, template_masks, template_colors):
        grid[r : r + 3, c : c + 3][mask] = int(color)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
