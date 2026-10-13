from __future__ import annotations

import random
from collections import Counter

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_OPEN_VECTORS = {
    "nw": (-1, -1),
    "ne": (-1, 1),
    "sw": (1, -1),
    "se": (1, 1),
}
_SOURCE_TEMPLATES = (
    np.array(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
)

_GRID_SIDE_MIN = 12
_GRID_SIDE_MAX = 30
_OBJECT_COUNT_RANGE = (2, 5)
_PLACEMENT_GAP = 1
_GEN_RETRIES = 400
_LAYOUT_RETRIES = 80
_MAX_OPTIONS_PER_OBJECT = 180


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            colors = Counter()

            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                colors[int(grid[r, c])] += 1

                for dr, dc in _NEIGHBORS4:
                    nr = int(r) + dr
                    nc = int(c) + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            assert len(colors) == 1, "each fragment must be monochrome"
            rs = [r for r, _ in coords]
            cs = [c for _, c in coords]
            components.append(
                {
                    "color": next(iter(colors)),
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                }
            )

    return components


def _mask_from_component(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _missing_corner(mask: np.ndarray) -> str:
    h, w = map(int, mask.shape)
    assert h == w, "L fragment bbox must be square"

    corners = {
        "nw": bool(mask[0, 0]),
        "ne": bool(mask[0, w - 1]),
        "sw": bool(mask[h - 1, 0]),
        "se": bool(mask[h - 1, w - 1]),
    }
    missing = [name for name, occupied in corners.items() if not occupied]
    assert len(missing) == 1, "expected exactly one missing corner"
    return missing[0]


def _reflect_mask(mask: np.ndarray, missing: str) -> np.ndarray:
    h, w = map(int, mask.shape)
    assert h == w
    out = np.zeros_like(mask)
    for r, c in np.argwhere(mask):
        rr = int(c)
        cc = int(r)
        if missing in {"ne", "sw"}:
            rr = h - 1 - int(c)
            cc = w - 1 - int(r)
        out[rr, cc] = True
    return out


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = map(int, mask.shape)
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _bbox_from_origin(r0: int, c0: int, h: int, w: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _rect_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = _PLACEMENT_GAP
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _dest_origin(mask: np.ndarray, r0: int, c0: int) -> tuple[int, int]:
    missing = _missing_corner(mask)
    dr, dc = _OPEN_VECTORS[missing]
    h, w = map(int, mask.shape)
    return int(r0 + dr * (h + 1)), int(c0 + dc * (w + 1))


def _fits(h: int, w: int, r0: int, c0: int, H: int, W: int) -> bool:
    return 0 <= r0 and 0 <= c0 and r0 + h <= H and c0 + w <= W


def main(grid: np.ndarray) -> np.ndarray:
    """Recognize rotated L fragments and place a mirrored copy one gap away in each fragment's opening direction.

    Input structure:
    - The grid has a dominant background color.
    - Several small monochrome foreground fragments are scattered apart from one another.
    - Each fragment is a 4-connected rotation of the same asymmetric L-like template inside a square bbox, with exactly one bbox corner missing.

    Rule:
    - For each fragment, detect which bbox corner is missing to recover the fragment's opening direction.
    - Reflect the fragment across the diagonal axis that points through that missing corner.
    - Keep the original fragment, then place the mirrored copy one empty cell away in that same diagonal direction.
    - If the mirrored copy would leave the grid, skip that copy.

    Key insight:
    - The only orientation cue is the missing bbox corner: it simultaneously determines the diagonal mirror axis and the diagonal translation direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = map(int, g.shape)
    bg = _get_bg(g)
    out = g.copy()

    components = _find_components(g, bg)
    assert components, "expected at least one foreground fragment"

    for comp in components:
        mask = _mask_from_component(comp)
        missing = _missing_corner(mask)
        mirrored = _reflect_mask(mask, missing)

        r0, c0, _r1, _c1 = comp["bbox"]
        nr0, nc0 = _dest_origin(mask, int(r0), int(c0))
        h, w = map(int, mask.shape)
        if not _fits(h, w, nr0, nc0, H, W):
            continue

        target = out[nr0 : nr0 + h, nc0 : nc0 + w]
        conflict = mirrored & (target != bg) & (target != int(comp["color"]))
        assert not bool(conflict.any()), "mirrored copies must land on empty space"
        target[mirrored] = int(comp["color"])

    return out


def _candidate_layouts(
    H: int, W: int, masks: list[np.ndarray], placeable: list[bool]
) -> list[list[tuple[int, int, tuple[int, int, int, int] | None]]]:
    all_options: list[list[tuple[int, int, tuple[int, int, int, int] | None]]] = []
    for mask, need_placeable in zip(masks, placeable):
        h, w = map(int, mask.shape)
        options: list[tuple[int, int, tuple[int, int, int, int] | None]] = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                nr0, nc0 = _dest_origin(mask, r0, c0)
                dest_ok = _fits(h, w, nr0, nc0, H, W)
                if dest_ok != need_placeable:
                    continue
                dest_bbox = _bbox_from_origin(nr0, nc0, h, w) if dest_ok else None
                options.append((int(r0), int(c0), dest_bbox))
        all_options.append(options)
    return all_options


def _choose_layout(
    H: int, W: int, masks: list[np.ndarray], placeable: list[bool]
) -> list[dict] | None:
    options = _candidate_layouts(H, W, masks, placeable)
    if any(not item for item in options):
        return None

    order = list(range(len(masks)))
    order.sort(key=lambda i: (len(options[i]), 0 if placeable[i] else 1))

    for _ in range(_LAYOUT_RETRIES):
        reserved: list[tuple[int, int, int, int]] = []
        layout: list[dict | None] = [None] * len(masks)
        ok = True

        for idx in order:
            local_options = options[idx][:]
            random.shuffle(local_options)
            if len(local_options) > _MAX_OPTIONS_PER_OBJECT:
                local_options = local_options[:_MAX_OPTIONS_PER_OBJECT]

            mask = masks[idx]
            h, w = map(int, mask.shape)
            picked = None
            for r0, c0, dest_bbox in local_options:
                src_bbox = _bbox_from_origin(r0, c0, h, w)
                rects = [src_bbox]
                if dest_bbox is not None:
                    rects.append(dest_bbox)
                if all(_rect_separated(rect, used) for rect in rects for used in reserved):
                    picked = {
                        "src_r0": int(r0),
                        "src_c0": int(c0),
                        "src_bbox": src_bbox,
                        "dst_bbox": dest_bbox,
                    }
                    break

            if picked is None:
                ok = False
                break

            layout[idx] = picked
            reserved.append(picked["src_bbox"])
            if picked["dst_bbox"] is not None:
                reserved.append(picked["dst_bbox"])

        if ok and all(item is not None for item in layout):
            return [item for item in layout if item is not None]

    return None


def _build_masks(count: int) -> list[np.ndarray]:
    base = random.choice(_SOURCE_TEMPLATES)
    return [np.rot90(base, random.randrange(4)).copy() for _ in range(count)]


def _branch_flags(count: int) -> list[bool]:
    flags = [random.random() < 0.5 for _ in range(count)]
    if all(flags):
        flags[random.randrange(count)] = False
    if not any(flags):
        flags[random.randrange(count)] = True
    return flags


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*_OBJECT_COUNT_RANGE)
    H = random.randint(_GRID_SIDE_MIN, _GRID_SIDE_MAX)
    W = random.randint(_GRID_SIDE_MIN, _GRID_SIDE_MAX)
    bg = random.randrange(10)

    palette = [color for color in range(10) if color != bg]
    if count > len(palette):
        return None
    colors = random.sample(palette, count)

    masks = _build_masks(count)
    placeable = _branch_flags(count)
    layout = _choose_layout(H, W, masks, placeable)
    if layout is None:
        return None

    inp = np.full((H, W), bg, dtype=int)
    expected = np.full((H, W), bg, dtype=int)

    for color, mask, can_place, placement in zip(colors, masks, placeable, layout):
        src_r0 = int(placement["src_r0"])
        src_c0 = int(placement["src_c0"])
        _paint_mask(inp, mask, src_r0, src_c0, color)
        _paint_mask(expected, mask, src_r0, src_c0, color)

        if can_place:
            missing = _missing_corner(mask)
            mirrored = _reflect_mask(mask, missing)
            dst_r0, dst_c0 = _dest_origin(mask, src_r0, src_c0)
            _paint_mask(expected, mirrored, dst_r0, dst_c0, color)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
