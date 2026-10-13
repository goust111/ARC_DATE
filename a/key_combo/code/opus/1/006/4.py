from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 4
MARKER = 0
RED = 2
GRAY = 5
RIGHT_COLOR = 3
LEFT_COLOR = 9

MAX_SIDE = 30
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, box


def _has_flanked_boundary_gap(mask: np.ndarray) -> bool:
    h, w = mask.shape

    top = mask[0, :]
    bottom = mask[h - 1, :]
    for c in range(w):
        if not top[c] and top[:c].any() and top[c + 1 :].any():
            return True
        if h > 1 and not bottom[c] and bottom[:c].any() and bottom[c + 1 :].any():
            return True

    left = mask[:, 0]
    right = mask[:, w - 1]
    for r in range(h):
        if not left[r] and left[:r].any() and left[r + 1 :].any():
            return True
        if w > 1 and not right[r] and right[:r].any() and right[r + 1 :].any():
            return True

    return False


def _find_marker(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    assert int(np.count_nonzero(g == MARKER)) == 4, "marker must be the only black content"

    hits = []
    for r in range(g.shape[0] - 1):
        for c in range(g.shape[1] - 1):
            if np.all(g[r : r + 2, c : c + 2] == MARKER):
                hits.append((r, c))
    assert len(hits) == 1, "expected a unique 2x2 black marker"
    return hits[0]


def _cross_mask(shape: tuple[int, int], marker_rc: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r0, c0 = marker_rc
    mask = np.zeros((h, w), dtype=bool)
    mask[r0 : r0 + 2, :] = True
    mask[:, c0 : c0 + 2] = True
    return mask


def _erase_cross(grid: np.ndarray, *, protect_red: bool) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    marker_rc = _find_marker(g)
    cross = _cross_mask(g.shape, marker_rc)

    out = g.copy()
    erase = cross & (out != BG)
    if protect_red:
        erase &= out != RED
    out[erase] = BG
    r0, c0 = marker_rc
    out[r0 : r0 + 2, c0 : c0 + 2] = BG
    return out, cross, marker_rc


def _foreground_components(grid: np.ndarray, *, include_marker: bool) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mask = g != BG
    if not include_marker:
        mask &= g != MARKER

    items = []
    for coords in _components_from_mask(mask):
        tight, box = _tight_mask(coords)
        items.append(
            {
                "coords": coords,
                "coord_set": {tuple(map(int, cell)) for cell in coords.tolist()},
                "bbox": box,
                "mask": tight,
                "has_gap": _has_flanked_boundary_gap(tight),
                "size": int(coords.shape[0]),
            }
        )
    return items


def _render_components(shape: tuple[int, int], components: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    w_grid = shape[1]
    for comp in components:
        r0, _c0, r1, _c1 = comp["bbox"]
        mask = comp["mask"]
        h, w = mask.shape
        dst_c = w_grid - w if comp["has_gap"] else 0
        color = RIGHT_COLOR if comp["has_gap"] else LEFT_COLOR
        dst = out[r0 : r1 + 1, dst_c : dst_c + w]
        assert dst.shape == (h, w)
        assert np.all(dst[mask] == BG), "translated components must not overlap"
        dst[mask] = color
    return out


def _post_components(grid: np.ndarray, *, protect_red: bool) -> list[dict]:
    erased, _cross, _marker = _erase_cross(grid, protect_red=protect_red)
    return _foreground_components(erased, include_marker=False)


def _solve_without_reanalysis(grid: np.ndarray) -> np.ndarray:
    original = _foreground_components(grid, include_marker=False)
    return _render_components(grid.shape, original)


def _solve_without_red_protection(grid: np.ndarray) -> np.ndarray:
    return _render_components(grid.shape, _post_components(grid, protect_red=False))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase a marker-defined cross, then slide each surviving component according to whether the erasure leaves a boundary notch.

    Input structure:
    - Yellow is the background and a unique 2x2 black marker identifies two rows and two columns.
    - The foreground consists of 5-7 separate 4-connected gray/red objects.
    - Red pixels belong to the same objects as gray pixels, but they are protected inside the marker cross.

    Rule:
    - Take the marker's two rows and two columns as one cross-shaped erase zone.
    - In that zone, erase every non-yellow, non-red pixel to yellow, then remove the 2x2 marker itself.
    - Recompute all 4-connected non-background components after the erasure.
    - If a component's bbox border contains a background gap flanked by component pixels on that same border, slide it to the right edge and recolor it green; otherwise slide it to the left edge and recolor it brown.
    - Perform the sliding on a blank yellow canvas.

    Key insight:
    - The slide direction is not an input property of the original object; it depends on the post-erasure component shape, and red pixels can preserve or destroy the decisive boundary structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= MAX_SIDE, "ARC grids must stay within 30x30"

    components = _post_components(g, protect_red=True)
    return _render_components(g.shape, components)


def _shape(arr: list[list[int]]) -> np.ndarray:
    return np.array(arr, dtype=int)


MANDATORY_CLOSED_TO_GAP = (
    {"array": _shape([[GRAY, GRAY, GRAY, GRAY], [GRAY, RED, RED, GRAY]]), "cross_offset": 1},
)
MANDATORY_PRE_GAP = (
    {"array": _shape([[GRAY, BG, BG, GRAY], [GRAY, RED, RED, GRAY]]), "cross_offset": 1},
)
MANDATORY_SPLITTER = (
    {"array": _shape([[GRAY, GRAY, BG, BG], [BG, GRAY, GRAY, BG], [BG, BG, GRAY, GRAY]]), "cross_offset": 1},
)
MANDATORY_CROSSED_STABLE = (
    {"array": _shape([[GRAY, RED, GRAY]]), "cross_offset": 1},
    {"array": _shape([[RED, GRAY], [RED, GRAY]]), "cross_offset": 0},
)

UNTOUCHED_LIBRARY = (
    _shape([[GRAY, GRAY], [GRAY, GRAY]]),
    _shape([[GRAY, BG], [GRAY, GRAY]]),
    _shape([[GRAY, BG, GRAY], [GRAY, GRAY, GRAY]]),
    _shape([[GRAY, GRAY, GRAY]]),
    _shape([[GRAY, GRAY, BG], [BG, GRAY, GRAY]]),
)


def _random_flips(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def _cross_offset_after_hflip(width: int, cross_offset: int) -> int:
    return width - cross_offset - 2


def _make_role(role: str) -> dict:
    if role == "closed_to_gap":
        spec = random.choice(MANDATORY_CLOSED_TO_GAP)
    elif role == "pre_gap":
        spec = random.choice(MANDATORY_PRE_GAP)
    elif role == "splitter":
        spec = random.choice(MANDATORY_SPLITTER)
    elif role == "crossed_stable":
        spec = random.choice(MANDATORY_CROSSED_STABLE)
    elif role == "untouched":
        return {"role": role, "crossed": False, "array": _random_flips(random.choice(UNTOUCHED_LIBRARY))}
    else:
        raise ValueError(role)

    arr = spec["array"].copy()
    cross_offset = int(spec["cross_offset"])
    if random.random() < 0.5:
        arr = np.flipud(arr)
    if random.random() < 0.5:
        arr = np.fliplr(arr)
        cross_offset = _cross_offset_after_hflip(arr.shape[1], cross_offset)
    return {"role": role, "crossed": True, "array": arr, "cross_offset": cross_offset}


def _paste_shape(grid: np.ndarray, shape: np.ndarray, top: int, left: int) -> None:
    h, w = shape.shape
    assert 0 <= top and 0 <= left and top + h <= grid.shape[0] and left + w <= grid.shape[1]
    patch = grid[top : top + h, left : left + w]
    mask = shape != BG
    assert np.all(patch[mask] == BG), "objects must not overlap"
    patch[mask] = shape[mask]


def _place_untouched_left(width: int, cross_col: int, shape_w: int) -> list[int]:
    hi = cross_col - shape_w - 1
    return list(range(0, hi + 1)) if hi >= 0 else []


def _place_untouched_right(width: int, cross_col: int, shape_w: int) -> list[int]:
    lo = cross_col + 3
    hi = width - shape_w
    return list(range(lo, hi + 1)) if lo <= hi else []


def _pre_label_map(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    objects = _foreground_components(grid, include_marker=False)
    labels = np.zeros(grid.shape, dtype=int)
    for idx, obj in enumerate(objects, start=1):
        labels[obj["coords"][:, 0], obj["coords"][:, 1]] = idx
    return objects, labels


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        correct = main(grid)
    except Exception:
        return False

    if max(grid.shape) > MAX_SIDE or max(correct.shape) > MAX_SIDE:
        return False

    try:
        marker_rc = _find_marker(grid)
        cross = _cross_mask(grid.shape, marker_rc)
        pre_objects, labels = _pre_label_map(grid)
        if not (5 <= len(pre_objects) <= 7):
            return False
        if any(not (3 <= obj["size"] <= 8) for obj in pre_objects):
            return False

        touched_ids = []
        untouched_ids = []
        for idx, obj in enumerate(pre_objects, start=1):
            coords = obj["coords"]
            touched = bool(np.any(cross[coords[:, 0], coords[:, 1]]))
            if touched:
                touched_ids.append(idx)
            else:
                untouched_ids.append(idx)
        if len(touched_ids) < 3 or not untouched_ids:
            return False

        erased, _cross, _marker = _erase_cross(grid, protect_red=True)
        post_objects = _foreground_components(erased, include_marker=False)
        if not post_objects:
            return False
        if not any(obj["has_gap"] for obj in post_objects):
            return False
        if not any(not obj["has_gap"] for obj in post_objects):
            return False

        mapping: dict[int, list[dict]] = {idx: [] for idx in range(1, len(pre_objects) + 1)}
        for obj in post_objects:
            ids = {int(labels[r, c]) for r, c in obj["coords"]}
            ids.discard(0)
            if len(ids) != 1:
                return False
            mapping[ids.pop()].append(obj)

        if len(post_objects) <= len(pre_objects):
            return False
        if not any(len(mapping[idx]) > 1 for idx in touched_ids):
            return False

        if not any(
            (not pre_objects[idx - 1]["has_gap"]) and any(post["has_gap"] for post in mapping[idx])
            for idx in touched_ids
        ):
            return False

        if not any(
            pre_objects[idx - 1]["has_gap"]
            and any(grid[r, c] == RED and cross[r, c] for r, c in pre_objects[idx - 1]["coord_set"])
            for idx in touched_ids
        ):
            return False

        if not any(
            len(mapping[idx]) == 1 and mapping[idx][0]["coord_set"] == pre_objects[idx - 1]["coord_set"]
            for idx in untouched_ids
        ):
            return False

        if int(np.count_nonzero(correct != BG)) != int(np.count_nonzero(erased != BG)):
            return False

        try:
            wrong_reanalysis = _solve_without_reanalysis(grid)
        except Exception:
            wrong_reanalysis = None
        try:
            wrong_red = _solve_without_red_protection(grid)
        except Exception:
            wrong_red = None

        if wrong_reanalysis is not None and np.array_equal(correct, wrong_reanalysis):
            return False
        if wrong_red is not None and np.array_equal(correct, wrong_red):
            return False
    except Exception:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    roles = [
        _make_role("closed_to_gap"),
        _make_role("pre_gap"),
        _make_role("splitter"),
        _make_role("crossed_stable"),
        _make_role("untouched"),
    ]
    for _ in range(random.randint(0, 2)):
        roles.append(_make_role("untouched"))
    random.shuffle(roles)

    marker_slot = random.randint(1, len(roles) - 1)
    sequence = roles[:marker_slot] + [{"role": "marker", "crossed": False, "array": np.zeros((2, 2), dtype=int)}] + roles[marker_slot:]

    top_margin = random.randint(0, 1)
    bottom_margin = random.randint(0, 1)
    tops = []
    row = top_margin
    for idx, item in enumerate(sequence):
        tops.append(row)
        row += item["array"].shape[0]
        if idx + 1 < len(sequence):
            row += 1 + random.randint(0, 1)
    height = row + bottom_margin
    if height > MAX_SIDE:
        return None

    width = random.randint(18, 26)
    cross_col = random.randint(6, width - 7)

    grid = np.full((height, width), BG, dtype=int)

    for item, top in zip(sequence, tops):
        shape = item["array"]
        if item["role"] == "marker":
            _paste_shape(grid, shape, top, cross_col)
            grid[top : top + 2, cross_col : cross_col + 2] = MARKER
            continue

        if item["crossed"]:
            left = cross_col - int(item["cross_offset"])
            if left < 0 or left + shape.shape[1] > width:
                return None
        else:
            left_choices = []
            left_choices.extend(_place_untouched_left(width, cross_col, shape.shape[1]))
            right_choices = _place_untouched_right(width, cross_col, shape.shape[1])
            if right_choices:
                left_choices.extend(right_choices)
            if not left_choices:
                return None
            left = random.choice(left_choices)

        _paste_shape(grid, shape, top, left)

    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
