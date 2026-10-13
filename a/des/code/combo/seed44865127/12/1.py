from __future__ import annotations

import random

import numpy as np


DIRS = ("up", "down", "left", "right")
MAX_GENERATE_ATTEMPTS = 400
PLACEMENT_GAP = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _foreground_colors(grid: np.ndarray, bg: int) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != int(bg)]


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _direction_from_missing(mask: np.ndarray) -> str:
    rows, cols = np.nonzero(mask)
    assert len(rows) > 0, "incomplete objects must have at least one missing cell"
    h, w = mask.shape
    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    row_bias = float(rows.mean()) - center_r
    col_bias = float(cols.mean()) - center_c
    row_score = abs(row_bias) / max(0.5, (h - 1) / 2.0)
    col_score = abs(col_bias) / max(0.5, (w - 1) / 2.0)

    if row_score > col_score + 1e-9:
        return "down" if row_bias > 0 else "up"
    if col_score > row_score + 1e-9:
        return "right" if col_bias > 0 else "left"
    raise AssertionError("missing-mask centroid direction is ambiguous")


def _shadow_shift(direction: str, h: int, w: int) -> tuple[int, int]:
    if direction == "up":
        return h, 0
    if direction == "down":
        return -h, 0
    if direction == "left":
        return 0, w
    if direction == "right":
        return 0, -w
    raise ValueError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify rectangles by completeness and cast each incomplete shape's missing
    mask as a same-color shadow on the opposite side.

    Input structure
    - The background is the dominant color.
    - Each foreground color forms one rectangular-bbox object.
    - Complete objects fully fill their bbox; incomplete ones contain only their
      own color or background inside that bbox.

    Rule
    - Complete rectangles stay unchanged.
    - For each incomplete rectangle, find the background-hole mask inside its
      bbox, determine whether that mask is biased up/down/left/right by its
      centroid, then copy the missing mask one bbox-height or bbox-width in the
      opposite direction using the object's color.
    - The original incomplete object remains unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    h_grid, w_grid = g.shape

    for color in _foreground_colors(g, bg):
        coords = np.argwhere(g == color)
        assert len(coords) > 0
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((region == bg) | (region == color)), "bbox must contain only object color or background"

        missing = region == bg
        if not np.any(missing):
            continue

        obj_h, obj_w = region.shape
        direction = _direction_from_missing(missing)
        shift_r, shift_c = _shadow_shift(direction, obj_h, obj_w)
        sr0 = r0 + shift_r
        sc0 = c0 + shift_c
        sr1 = sr0 + obj_h - 1
        sc1 = sc0 + obj_w - 1
        assert 0 <= sr0 <= sr1 < h_grid and 0 <= sc0 <= sc1 < w_grid, "shadow must stay in bounds"

        in_patch = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
        out_patch = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
        assert np.all(in_patch[missing] == bg), "shadow target cells must be empty in the input"
        assert np.all(out_patch[missing] == bg), "shadows must not overlap each other"
        out_patch = out_patch.copy()
        out_patch[missing] = color
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = out_patch

    return out


def _box_conflicts(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _bool_component_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _bbox_is_preserved(missing: np.ndarray) -> bool:
    filled = ~missing
    return bool(
        filled[0, :].any()
        and filled[-1, :].any()
        and filled[:, 0].any()
        and filled[:, -1].any()
    )


def _centered_start(length: int, span: int) -> int:
    if span >= length:
        return 0
    preferred = (length - span) // 2
    wiggle = max(0, (length - span) // 3)
    lo = max(0, preferred - wiggle)
    hi = min(length - span, preferred + wiggle)
    return random.randint(lo, hi)


def _make_missing_mask(obj_h: int, obj_w: int, direction: str) -> np.ndarray:
    for _ in range(200):
        missing = np.zeros((obj_h, obj_w), dtype=bool)
        if direction in ("up", "down"):
            bite_h = random.randint(1, max(1, obj_h // 2))
            bite_w = random.randint(1, obj_w - 1)
            c0 = _centered_start(obj_w, bite_w)
            if direction == "up":
                missing[:bite_h, c0 : c0 + bite_w] = True
                if random.random() < 0.65 and bite_h < obj_h - 1:
                    ext_h = random.randint(1, min(max(1, obj_h // 2), obj_h - bite_h))
                    ext_w = random.randint(1, bite_w)
                    ext_c0 = random.randint(c0, c0 + bite_w - ext_w)
                    missing[bite_h : bite_h + ext_h, ext_c0 : ext_c0 + ext_w] = True
            else:
                missing[obj_h - bite_h :, c0 : c0 + bite_w] = True
                if random.random() < 0.65 and bite_h < obj_h - 1:
                    ext_h = random.randint(1, min(max(1, obj_h // 2), obj_h - bite_h))
                    ext_w = random.randint(1, bite_w)
                    ext_c0 = random.randint(c0, c0 + bite_w - ext_w)
                    missing[obj_h - bite_h - ext_h : obj_h - bite_h, ext_c0 : ext_c0 + ext_w] = True
        else:
            bite_h = random.randint(1, obj_h - 1)
            bite_w = random.randint(1, max(1, obj_w // 2))
            r0 = _centered_start(obj_h, bite_h)
            if direction == "left":
                missing[r0 : r0 + bite_h, :bite_w] = True
                if random.random() < 0.65 and bite_w < obj_w - 1:
                    ext_w = random.randint(1, min(max(1, obj_w // 2), obj_w - bite_w))
                    ext_h = random.randint(1, bite_h)
                    ext_r0 = random.randint(r0, r0 + bite_h - ext_h)
                    missing[ext_r0 : ext_r0 + ext_h, bite_w : bite_w + ext_w] = True
            else:
                missing[r0 : r0 + bite_h, obj_w - bite_w :] = True
                if random.random() < 0.65 and bite_w < obj_w - 1:
                    ext_w = random.randint(1, min(max(1, obj_w // 2), obj_w - bite_w))
                    ext_h = random.randint(1, bite_h)
                    ext_r0 = random.randint(r0, r0 + bite_h - ext_h)
                    missing[ext_r0 : ext_r0 + ext_h, obj_w - bite_w - ext_w : obj_w - bite_w] = True

        if not _bbox_is_preserved(missing):
            continue
        if not _bool_component_connected(~missing):
            continue
        try:
            inferred = _direction_from_missing(missing)
        except AssertionError:
            continue
        if inferred == direction:
            return missing

    raise RuntimeError("failed to create an unambiguous missing mask")


def _build_specs(count: int) -> list[dict[str, object]]:
    bg = random.randint(0, 9)
    colors = random.sample([c for c in range(10) if c != bg], count)
    incomplete_count = random.randint(1, count - 1)
    statuses = [True] * incomplete_count + [False] * (count - incomplete_count)
    random.shuffle(statuses)
    direction_pool = list(DIRS) * ((incomplete_count + len(DIRS) - 1) // len(DIRS))
    random.shuffle(direction_pool)
    direction_pool = direction_pool[:incomplete_count]

    specs: list[dict[str, object]] = []
    size_hi = 6 if count >= 4 else 7
    incomplete_seen = 0
    for color, incomplete in zip(colors, statuses):
        obj_h = random.randint(4, size_hi)
        obj_w = random.randint(4, size_hi)
        if incomplete:
            direction = direction_pool[incomplete_seen]
            incomplete_seen += 1
            missing = _make_missing_mask(obj_h, obj_w, direction)
            shift = _shadow_shift(direction, obj_h, obj_w)
        else:
            direction = None
            missing = np.zeros((obj_h, obj_w), dtype=bool)
            shift = (0, 0)
        specs.append(
            {
                "color": int(color),
                "incomplete": bool(incomplete),
                "direction": direction,
                "missing": missing,
                "shape": (obj_h, obj_w),
                "shift": shift,
            }
        )
    return bg, specs


def _boxes_for_position(spec: dict[str, object], r0: int, c0: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int] | None, tuple[int, int, int, int]]:
    obj_h, obj_w = spec["shape"]
    shift_r, shift_c = spec["shift"]
    bbox = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
    if not spec["incomplete"]:
        return bbox, None, bbox
    sr0 = r0 + shift_r
    sc0 = c0 + shift_c
    shadow = (sr0, sc0, sr0 + obj_h - 1, sc0 + obj_w - 1)
    union = (
        min(bbox[0], shadow[0]),
        min(bbox[1], shadow[1]),
        max(bbox[2], shadow[2]),
        max(bbox[3], shadow[3]),
    )
    return bbox, shadow, union


def _candidate_positions(spec: dict[str, object], grid_h: int, grid_w: int) -> list[tuple[int, int]]:
    obj_h, obj_w = spec["shape"]
    shift_r, shift_c = spec["shift"]
    rmin = max(0, -shift_r)
    rmax = min(grid_h - obj_h, grid_h - obj_h - shift_r)
    cmin = max(0, -shift_c)
    cmax = min(grid_w - obj_w, grid_w - obj_w - shift_c)
    if rmin > rmax or cmin > cmax:
        return []
    positions = [(r, c) for r in range(rmin, rmax + 1) for c in range(cmin, cmax + 1)]
    random.shuffle(positions)
    return positions


def _place_specs(specs: list[dict[str, object]], grid_h: int, grid_w: int) -> list[dict[str, tuple[int, int, int, int] | None]] | None:
    order = sorted(
        range(len(specs)),
        key=lambda i: (
            specs[i]["incomplete"],
            specs[i]["shape"][0] * specs[i]["shape"][1],
        ),
        reverse=True,
    )
    candidate_map = {i: _candidate_positions(specs[i], grid_h, grid_w) for i in range(len(specs))}
    if any(not candidate_map[i] for i in range(len(specs))):
        return None

    placements: list[dict[str, tuple[int, int, int, int] | None] | None] = [None] * len(specs)
    used_unions: list[tuple[int, int, int, int]] = []

    def rec(depth: int) -> bool:
        if depth == len(order):
            return True
        idx = order[depth]
        spec = specs[idx]
        for r0, c0 in candidate_map[idx]:
            bbox, shadow, union = _boxes_for_position(spec, r0, c0)
            if any(_box_conflicts(union, other) for other in used_unions):
                continue
            placements[idx] = {"bbox": bbox, "shadow": shadow, "union": union}
            used_unions.append(union)
            if rec(depth + 1):
                return True
            used_unions.pop()
            placements[idx] = None
        return False

    if not rec(0):
        return None
    return [placement for placement in placements if placement is not None]


def _render_scene(grid_h: int, grid_w: int, bg: int, specs: list[dict[str, object]], placements: list[dict[str, tuple[int, int, int, int] | None]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    out = np.full((grid_h, grid_w), int(bg), dtype=int)

    for spec, placement in zip(specs, placements):
        color = int(spec["color"])
        obj_h, obj_w = spec["shape"]
        missing = np.asarray(spec["missing"], dtype=bool)
        r0, c0, _, _ = placement["bbox"]

        patch = np.full((obj_h, obj_w), color, dtype=int)
        patch[missing] = bg
        inp[r0 : r0 + obj_h, c0 : c0 + obj_w] = patch
        out[r0 : r0 + obj_h, c0 : c0 + obj_w] = patch

        if placement["shadow"] is not None:
            sr0, sc0, _, _ = placement["shadow"]
            shadow_patch = out[sr0 : sr0 + obj_h, sc0 : sc0 + obj_w].copy()
            shadow_patch[missing] = color
            out[sr0 : sr0 + obj_h, sc0 : sc0 + obj_w] = shadow_patch

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(18, 26)
    grid_w = random.randint(18, 26)
    count = random.randint(2, 4)
    bg, specs = _build_specs(count)
    placements = _place_specs(specs, grid_h, grid_w)
    if placements is None:
        return None

    inp, expected = _render_scene(grid_h, grid_w, bg, specs, placements)
    if _get_bg(inp) != bg:
        return None
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
