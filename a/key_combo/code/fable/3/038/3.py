from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
GREY = 5


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    cropped = _crop_bool(mask)
    return tuple(map(int, cropped.shape)), cropped.astype(np.uint8).tobytes()


def _shape_orbit(mask: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    cropped = _crop_bool(mask)
    return {
        _mask_signature(cropped),
        _mask_signature(np.fliplr(cropped)),
        _mask_signature(np.flipud(cropped)),
        _mask_signature(np.rot90(cropped, 2)),
    }


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    H, W = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(mask.sum())


def _find_components(mask: np.ndarray) -> list[dict]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0, r1, c1 = _bbox_coords(coords)
            local = coords - np.array([[r0, c0]])
            crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            crop[local[:, 0], local[:, 1]] = True
            comps.append(
                {
                    "coords": coords,
                    "coord_set": {(int(rr), int(cc)) for rr, cc in coords},
                    "bbox": (r0, c0, r1, c1),
                    "mask": crop,
                }
            )
    return comps


def _vertical_pair_axis(a: dict, b: dict) -> int | None:
    if a["mask"].shape != b["mask"].shape:
        return None
    if not np.array_equal(np.fliplr(a["mask"]), b["mask"]):
        return None
    _r0a, c0a, _r1a, c1a = a["bbox"]
    _r0b, c0b, _r1b, c1b = b["bbox"]
    axis2 = c0a + c1b
    if axis2 != c1a + c0b or axis2 % 2:
        return None
    reflected = {(r, axis2 - c) for r, c in a["coord_set"]}
    if reflected != b["coord_set"]:
        return None
    return axis2 // 2


def _horizontal_pair_axis(a: dict, b: dict) -> int | None:
    if a["mask"].shape != b["mask"].shape:
        return None
    if not np.array_equal(np.flipud(a["mask"]), b["mask"]):
        return None
    r0a, _c0a, r1a, _c1a = a["bbox"]
    r0b, _c0b, r1b, _c1b = b["bbox"]
    axis2 = r0a + r1b
    if axis2 != r1a + r0b or axis2 % 2:
        return None
    reflected = {(axis2 - r, c) for r, c in a["coord_set"]}
    if reflected != b["coord_set"]:
        return None
    return axis2 // 2


def _rotation_center(a: dict, b: dict) -> tuple[int, int] | None:
    if a["mask"].shape != b["mask"].shape:
        return None
    if not np.array_equal(np.rot90(a["mask"], 2), b["mask"]):
        return None
    r0a, c0a, r1a, c1a = a["bbox"]
    r0b, c0b, r1b, c1b = b["bbox"]
    center_r2 = r0a + r1b
    center_c2 = c0a + c1b
    if center_r2 != r1a + r0b or center_c2 != c1a + c0b:
        return None
    if center_r2 % 2 or center_c2 % 2:
        return None
    rotated = {(center_r2 - r, center_c2 - c) for r, c in a["coord_set"]}
    if rotated != b["coord_set"]:
        return None
    return center_r2 // 2, center_c2 // 2


def _self_vertical_axis(comp: dict) -> int | None:
    mask = comp["mask"]
    if mask.shape[1] % 2 == 0 or not np.array_equal(mask, np.fliplr(mask)):
        return None
    _r0, c0, _r1, c1 = comp["bbox"]
    axis2 = c0 + c1
    reflected = {(r, axis2 - c) for r, c in comp["coord_set"]}
    if reflected != comp["coord_set"]:
        return None
    return axis2 // 2


def _self_horizontal_axis(comp: dict) -> int | None:
    mask = comp["mask"]
    if mask.shape[0] % 2 == 0 or not np.array_equal(mask, np.flipud(mask)):
        return None
    r0, _c0, r1, _c1 = comp["bbox"]
    axis2 = r0 + r1
    reflected = {(axis2 - r, c) for r, c in comp["coord_set"]}
    if reflected != comp["coord_set"]:
        return None
    return axis2 // 2


def _solve_components(components: list[dict]) -> list[dict]:
    n = len(components)
    assert n % 2 == 1 and n > 0

    adjacency: dict[int, list[dict]] = {i: [] for i in range(n)}
    self_options: dict[int, list[dict]] = {i: [] for i in range(n)}
    for i, comp in enumerate(components):
        axis = _self_vertical_axis(comp)
        if axis is not None:
            self_options[i].append({"kind": "v", "members": (i,), "value": axis})
        axis = _self_horizontal_axis(comp)
        if axis is not None:
            self_options[i].append({"kind": "h", "members": (i,), "value": axis})

    for i in range(n):
        for j in range(i + 1, n):
            a = components[i]
            b = components[j]
            axis = _vertical_pair_axis(a, b)
            if axis is not None:
                option = {"kind": "v", "members": (i, j), "value": axis}
                adjacency[i].append(option)
                adjacency[j].append(option)

            axis = _horizontal_pair_axis(a, b)
            if axis is not None:
                option = {"kind": "h", "members": (i, j), "value": axis}
                adjacency[i].append(option)
                adjacency[j].append(option)

            center = _rotation_center(a, b)
            if center is not None:
                option = {"kind": "r", "members": (i, j), "value": center}
                adjacency[i].append(option)
                adjacency[j].append(option)

    solutions: list[list[dict]] = []

    def backtrack(remaining: tuple[int, ...], used_single: bool, chosen: list[dict]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            if used_single:
                solutions.append(chosen.copy())
            return

        i = remaining[0]
        rem_set = set(remaining)
        if not used_single:
            for option in self_options[i]:
                chosen.append(option)
                backtrack(tuple(x for x in remaining if x != i), True, chosen)
                chosen.pop()

        for option in adjacency[i]:
            _ii, j = option["members"]
            if j not in rem_set:
                continue
            next_remaining = tuple(x for x in remaining if x not in {i, j})
            chosen.append(option)
            backtrack(next_remaining, used_single, chosen)
            chosen.pop()

    backtrack(tuple(range(n)), False, [])
    assert len(solutions) == 1, "expected exactly one valid pairing solution"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark the hidden symmetry axis or rotation center for each uniquely matched same-color sprite pair.

    Input structure:
    - The grid has a black background and several 4-connected gray sprites.
    - Every sprite except one belongs to a unique pair of equally shaped sprites.
    - Each pair is related by exactly one whole-column mirror, whole-row mirror, or 180-degree rotation around one cell.
    - The lone leftover sprite is symmetric to itself across one whole column or one whole row.

    Rule:
    - Find the unique global pairing of the gray sprites by testing those symmetry relations.
    - For every whole-column mirror pair, paint the entire axis column green.
    - For every whole-row mirror pair, paint the entire axis row green.
    - For every 180-degree pair, paint only the rotation-center cell yellow.
    - For the lone self-symmetric sprite, paint its own symmetry axis green.

    Key insight:
    - Color gives no pairing information, so the task is to recover the only consistent shape matching under three different symmetries and then emit the implied hidden axis or center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _get_bg(g)
    components = _find_components(g != bg)
    solution = _solve_components(components)

    out = np.full(g.shape, BLACK, dtype=int)
    for option in solution:
        if option["kind"] == "v":
            out[:, int(option["value"])] = GREEN
        elif option["kind"] == "h":
            out[int(option["value"]), :] = GREEN
        else:
            r, c = option["value"]
            out[int(r), int(c)] = YELLOW
    return out


_GEN_ATTEMPTS = 400
_GRID_RANGE = (22, 30)
_BOX_PADDING = 2
_BORDER = 1


def _paint_mask(dst: np.ndarray, mask: np.ndarray, top: int, left: int, color: int = GREY) -> None:
    h, w = mask.shape
    view = dst[top : top + h, left : left + w]
    view[mask] = int(color)


def _random_connected_mask(
    min_h: int = 2,
    max_h: int = 5,
    min_w: int = 2,
    max_w: int = 5,
    min_cells: int = 4,
    max_cells: int = 8,
) -> np.ndarray:
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    target = random.randint(min_cells, min(max_cells, h * w))
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = {
        nbr
        for nbr in _neighbors4(*start)
        if 0 <= nbr[0] < h and 0 <= nbr[1] < w
    }
    while int(mask.sum()) < target:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nbr in _neighbors4(*cell):
            rr, cc = nbr
            if 0 <= rr < h and 0 <= cc < w and not mask[rr, cc]:
                frontier.add(nbr)
    return _crop_bool(mask)


def _is_asymmetric(mask: np.ndarray) -> bool:
    return (
        mask.shape[0] >= 2
        and mask.shape[1] >= 2
        and not np.array_equal(mask, np.fliplr(mask))
        and not np.array_equal(mask, np.flipud(mask))
        and not np.array_equal(mask, np.rot90(mask, 2))
    )


def _sample_asymmetric_mask(used: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(300):
        mask = _random_connected_mask()
        orbit = _shape_orbit(mask)
        if not _is_asymmetric(mask) or orbit & used:
            continue
        used.update(orbit)
        return mask
    return None


def _sample_vertical_self_mask(used: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(300):
        base = _random_connected_mask(min_h=3, max_h=6, min_w=2, max_w=4, min_cells=3, max_cells=7)
        mask = _crop_bool(base | np.fliplr(base))
        orbit = _shape_orbit(mask)
        if (
            mask.shape[1] < 3
            or mask.shape[1] % 2 == 0
            or int(mask.sum()) < 5
            or not _is_connected(mask)
            or not np.array_equal(mask, np.fliplr(mask))
            or np.array_equal(mask, np.flipud(mask))
            or orbit & used
        ):
            continue
        used.update(orbit)
        return mask
    return None


def _sample_horizontal_self_mask(used: set[tuple[tuple[int, int], bytes]]) -> np.ndarray | None:
    for _ in range(300):
        base = _random_connected_mask(min_h=2, max_h=4, min_w=3, max_w=6, min_cells=3, max_cells=7)
        mask = _crop_bool(base | np.flipud(base))
        orbit = _shape_orbit(mask)
        if (
            mask.shape[0] < 3
            or mask.shape[0] % 2 == 0
            or int(mask.sum()) < 5
            or not _is_connected(mask)
            or not np.array_equal(mask, np.flipud(mask))
            or np.array_equal(mask, np.fliplr(mask))
            or orbit & used
        ):
            continue
        used.update(orbit)
        return mask
    return None


def _build_vertical_patch(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = mask.shape
    gap_half = random.randint(0, 1)
    axis_c = w + gap_half
    right_c0 = w + 1 + 2 * gap_half
    patch = np.zeros((h, right_c0 + w), dtype=int)
    _paint_mask(patch, mask, 0, 0)
    _paint_mask(patch, np.fliplr(mask), 0, right_c0)
    return patch, {"kind": "v", "offset": (0, axis_c)}


def _build_horizontal_patch(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = mask.shape
    gap_half = random.randint(0, 1)
    axis_r = h + gap_half
    bottom_r0 = h + 1 + 2 * gap_half
    patch = np.zeros((bottom_r0 + h, w), dtype=int)
    _paint_mask(patch, mask, 0, 0)
    _paint_mask(patch, np.flipud(mask), bottom_r0, 0)
    return patch, {"kind": "h", "offset": (axis_r, 0)}


def _build_rotation_patch(mask: np.ndarray) -> tuple[np.ndarray, dict]:
    h, w = mask.shape
    gap_r = random.randint(0, 1)
    gap_c = random.randint(0, 1)
    center_r = h + gap_r
    center_c = w + gap_c
    other_r0 = h + 1 + 2 * gap_r
    other_c0 = w + 1 + 2 * gap_c
    patch = np.zeros((other_r0 + h, other_c0 + w), dtype=int)
    _paint_mask(patch, mask, 0, 0)
    _paint_mask(patch, np.rot90(mask, 2), other_r0, other_c0)
    return patch, {"kind": "r", "offset": (center_r, center_c)}


def _build_singleton_patch(mask: np.ndarray, kind: str) -> tuple[np.ndarray, dict]:
    h, w = mask.shape
    patch = np.zeros((h, w), dtype=int)
    _paint_mask(patch, mask, 0, 0)
    if kind == "v":
        return patch, {"kind": "v", "offset": (0, w // 2)}
    return patch, {"kind": "h", "offset": (h // 2, 0)}


def _random_free_box(occupied: np.ndarray, box_h: int, box_w: int) -> tuple[int, int] | None:
    H, W = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(_BORDER, H - _BORDER - box_h + 1):
        for c0 in range(_BORDER, W - _BORDER - box_w + 1):
            rr0 = max(0, r0 - _BOX_PADDING)
            cc0 = max(0, c0 - _BOX_PADDING)
            rr1 = min(H, r0 + box_h + _BOX_PADDING)
            cc1 = min(W, c0 + box_w + _BOX_PADDING)
            if not occupied[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _emit_expected(shape: tuple[int, int], markers: list[dict]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    for marker in markers:
        if marker["kind"] == "v":
            out[:, marker["value"]] = GREEN
        elif marker["kind"] == "h":
            out[marker["value"], :] = GREEN
        else:
            r, c = marker["value"]
            out[r, c] = YELLOW
    return out


def _markers_are_clear(markers: list[dict]) -> bool:
    cols = [m["value"] for m in markers if m["kind"] == "v"]
    rows = [m["value"] for m in markers if m["kind"] == "h"]
    centers = [m["value"] for m in markers if m["kind"] == "r"]
    if len(cols) != len(set(cols)) or len(rows) != len(set(rows)) or len(centers) != len(set(centers)):
        return False
    col_set = set(cols)
    row_set = set(rows)
    for r, c in centers:
        if r in row_set or c in col_set:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_GRID_RANGE)
    W = random.randint(*_GRID_RANGE)
    used_orbits: set[tuple[tuple[int, int], bytes]] = set()

    masks: list[tuple[str, np.ndarray]] = []
    for kind in ("v", "h", "r"):
        mask = _sample_asymmetric_mask(used_orbits)
        if mask is None:
            return None
        masks.append((kind, mask))

    extra_kind = random.choice(["v", "h", "r", None, None])
    if extra_kind is not None:
        mask = _sample_asymmetric_mask(used_orbits)
        if mask is None:
            return None
        masks.append((extra_kind, mask))

    single_kind = random.choice(["v", "h"])
    if single_kind == "v":
        single_mask = _sample_vertical_self_mask(used_orbits)
    else:
        single_mask = _sample_horizontal_self_mask(used_orbits)
    if single_mask is None:
        return None

    units: list[dict] = []
    for kind, mask in masks:
        if kind == "v":
            patch, marker = _build_vertical_patch(mask)
        elif kind == "h":
            patch, marker = _build_horizontal_patch(mask)
        else:
            patch, marker = _build_rotation_patch(mask)
        units.append({"patch": patch, "marker": marker})

    patch, marker = _build_singleton_patch(single_mask, single_kind)
    units.append({"patch": patch, "marker": marker})

    order = sorted(range(len(units)), key=lambda i: units[i]["patch"].shape[0] * units[i]["patch"].shape[1], reverse=True)
    occupied = np.zeros((H, W), dtype=bool)
    grid = np.full((H, W), BLACK, dtype=int)
    markers: list[dict] = []
    for idx in order:
        patch = units[idx]["patch"]
        top_left = _random_free_box(occupied, patch.shape[0], patch.shape[1])
        if top_left is None:
            return None
        r0, c0 = top_left
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = True
        _paint_mask(grid, patch == GREY, r0, c0)
        marker = units[idx]["marker"]
        if marker["kind"] == "v":
            markers.append({"kind": "v", "value": c0 + marker["offset"][1]})
        elif marker["kind"] == "h":
            markers.append({"kind": "h", "value": r0 + marker["offset"][0]})
        else:
            markers.append({"kind": "r", "value": (r0 + marker["offset"][0], c0 + marker["offset"][1])})

    if not _markers_are_clear(markers):
        return None

    expected = _emit_expected((H, W), markers)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-pairing input")
