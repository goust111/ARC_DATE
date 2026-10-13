import random
from typing import Any

import numpy as np


BG = 0
MAROON = 9
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in (
                    (cr - 1, cc),
                    (cr + 1, cc),
                    (cr, cc - 1),
                    (cr, cc + 1),
                ):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _build_local_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _frame_info(
    grid: np.ndarray, color: int, points: list[tuple[int, int]]
) -> dict[str, Any] | None:
    del color
    mask, bbox = _build_local_mask(points)
    r0, c0, r1, c1 = bbox
    h, w = mask.shape
    if h < 3 or w < 3:
        return None

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None

    missing = border & ~mask
    if int(missing.sum()) != 1:
        return None

    mr, mc = map(int, np.argwhere(missing)[0])
    abs_r, abs_c = r0 + mr, c0 + mc
    if int(grid[abs_r, abs_c]) != MAROON:
        return None

    if mr == 0 and 0 < mc < w - 1 and w % 2 == 1 and mc == w // 2:
        side = "top"
    elif mr == h - 1 and 0 < mc < w - 1 and w % 2 == 1 and mc == w // 2:
        side = "bottom"
    elif mc == 0 and 0 < mr < h - 1 and h % 2 == 1 and mr == h // 2:
        side = "left"
    elif mc == w - 1 and 0 < mr < h - 1 and h % 2 == 1 and mr == h // 2:
        side = "right"
    else:
        return None

    expected_count = 2 * (h + w) - 4 - 1
    if len(points) != expected_count:
        return None

    return {
        "bbox": bbox,
        "side": side,
        "marker": (abs_r, abs_c),
    }


def _flip_key(mask: np.ndarray, side: str) -> np.ndarray:
    return np.fliplr(mask) if side in ("left", "right") else np.flipud(mask)


def _placement_origin(
    frame_bbox: tuple[int, int, int, int], side: str, key_shape: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    kh, kw = key_shape
    assert kh < inner_h and kw < inner_w

    if side == "left":
        return inner_r0 + (inner_h - kh) // 2, inner_c0
    if side == "right":
        return inner_r0 + (inner_h - kh) // 2, inner_c0 + inner_w - kw
    if side == "top":
        return inner_r0, inner_c0 + (inner_w - kw) // 2
    return inner_r0 + inner_h - kh, inner_c0 + (inner_w - kw) // 2


def _extract_pairs(grid: np.ndarray) -> list[dict[str, Any]]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, MAROON)]
    pairs: list[dict[str, Any]] = []
    for color in colors:
        comps = _components(grid == color)
        assert len(comps) >= 2, f"color {color} should form one frame and one key"

        frame_infos: list[dict[str, Any]] = []
        key_comps: list[list[tuple[int, int]]] = []
        for comp in comps:
            info = _frame_info(grid, color, comp)
            if info is None:
                key_comps.append(comp)
            else:
                frame_infos.append(info)

        assert len(frame_infos) == 1, f"color {color} should have exactly one frame"
        assert len(key_comps) == 1, f"color {color} should have exactly one key"

        key_points = key_comps[0]
        key_mask, _ = _build_local_mask(key_points)
        pairs.append(
            {
                "color": color,
                "frame_bbox": frame_infos[0]["bbox"],
                "side": frame_infos[0]["side"],
                "marker": frame_infos[0]["marker"],
                "key_points": key_points,
                "key_mask": key_mask,
            }
        )

    assert pairs, "expected at least one frame/key pair"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each colored key with the matching rectangular frame, mirror the key
    according to the maroon marker side, then insert it flush against that side.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for pair in _extract_pairs(g):
        color = pair["color"]
        for r, c in pair["key_points"]:
            out[r, c] = BG

        mr, mc = pair["marker"]
        out[mr, mc] = color

        placed_mask = _flip_key(pair["key_mask"], pair["side"])
        pr, pc = _placement_origin(pair["frame_bbox"], pair["side"], placed_mask.shape)
        rr, cc = np.nonzero(placed_mask)
        out[pr + rr, pc + cc] = color

    return out


def _random_path(
    start: tuple[int, int], end: tuple[int, int], rng: random.Random
) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]
    while (r, c) != (er, ec):
        choices: list[tuple[int, int]] = []
        if r < er:
            choices.append((r + 1, c))
        elif r > er:
            choices.append((r - 1, c))
        if c < ec:
            choices.append((r, c + 1))
        elif c > ec:
            choices.append((r, c - 1))
        r, c = rng.choice(choices)
        path.append((r, c))
    return path


def _generate_key_mask(kh: int, kw: int, side: str, rng: random.Random) -> np.ndarray | None:
    flip = np.fliplr if side in ("left", "right") else np.flipud
    min_area = max(4, (kh * kw + 2) // 3)
    max_area = max(min_area, kh * kw - 1)

    for _ in range(200):
        anchors = [
            (0, rng.randrange(kw)),
            (kh - 1, rng.randrange(kw)),
            (rng.randrange(kh), 0),
            (rng.randrange(kh), kw - 1),
        ]
        rng.shuffle(anchors)
        mask = np.zeros((kh, kw), dtype=bool)
        current = anchors[0]
        mask[current] = True
        for nxt in anchors[1:]:
            for cell in _random_path(current, nxt, rng):
                mask[cell] = True
            current = nxt

        current_area = int(mask.sum())
        if current_area > max_area:
            continue
        target = rng.randint(max(min_area, current_area), max_area)
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r in range(kh):
                for c in range(kw):
                    if mask[r, c]:
                        continue
                    if (
                        (r > 0 and mask[r - 1, c])
                        or (r + 1 < kh and mask[r + 1, c])
                        or (c > 0 and mask[r, c - 1])
                        or (c + 1 < kw and mask[r, c + 1])
                    ):
                        frontier.append((r, c))
            if not frontier:
                break
            r, c = rng.choice(frontier)
            mask[r, c] = True

        pts = np.argwhere(mask)
        if len(np.unique(pts[:, 0])) < 2 or len(np.unique(pts[:, 1])) < 2:
            continue
        if np.array_equal(mask, flip(mask)):
            continue
        if mask.all():
            continue
        return mask

    return None


def _make_frame_patch(color: int, h: int, w: int, side: str) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if side == "top":
        patch[0, w // 2] = MAROON
    elif side == "bottom":
        patch[h - 1, w // 2] = MAROON
    elif side == "left":
        patch[h // 2, 0] = MAROON
    else:
        patch[h // 2, w - 1] = MAROON
    return patch


def _make_key_patch(color: int, mask: np.ndarray) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _can_place(blocked: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = blocked.shape
    ph, pw = shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + ph + 1)
    c1 = min(w, left + pw + 1)
    return not blocked[r0:r1, c0:c1].any()


def _stamp(
    grid: np.ndarray, blocked: np.ndarray, top: int, left: int, patch: np.ndarray
) -> None:
    ph, pw = patch.shape
    mask = patch != BG
    grid[top : top + ph, left : left + pw][mask] = patch[mask]
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(grid.shape[0], top + ph + 1)
    c1 = min(grid.shape[1], left + pw + 1)
    blocked[r0:r1, c0:c1] = True


def _frame_size_options(pair_count: int) -> list[int]:
    return [7, 9, 11] if pair_count <= 3 else [7, 9]


def _choose_key_dims(
    inner_h: int, inner_w: int, side: str, rng: random.Random
) -> tuple[int, int] | None:
    kh_choices = [v for v in range(2, inner_h) if v >= 2]
    kw_choices = [v for v in range(2, inner_w) if v >= 2]
    if side in ("left", "right"):
        kh_choices = [v for v in kh_choices if (inner_h - v) % 2 == 0 and v >= 3]
    else:
        kw_choices = [v for v in kw_choices if (inner_w - v) % 2 == 0 and v >= 3]
    if not kh_choices or not kw_choices:
        return None
    return rng.choice(kh_choices), rng.choice(kw_choices)


def _spec_output(grid: np.ndarray, specs: list[dict[str, Any]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        color = spec["color"]
        key_patch = spec["key_patch"]
        kr, kc = spec["key_pos"]
        out[kr : kr + key_patch.shape[0], kc : kc + key_patch.shape[1]][key_patch == color] = BG

        fr, fc = spec["frame_pos"]
        fh, fw = spec["frame_patch"].shape
        if spec["side"] == "top":
            out[fr, fc + fw // 2] = color
        elif spec["side"] == "bottom":
            out[fr + fh - 1, fc + fw // 2] = color
        elif spec["side"] == "left":
            out[fr + fh // 2, fc] = color
        else:
            out[fr + fh // 2, fc + fw - 1] = color

        placed_mask = _flip_key(spec["key_mask"], spec["side"])
        frame_bbox = (fr, fc, fr + fh - 1, fc + fw - 1)
        pr, pc = _placement_origin(frame_bbox, spec["side"], placed_mask.shape)
        rr, cc = np.nonzero(placed_mask)
        out[pr + rr, pc + cc] = color

    return out


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = rng.randint(2, 4)
    size_min = 18 + 4 * (pair_count - 2)
    size_max = 24 + 4 * (pair_count - 2)
    h = rng.randint(size_min, size_max)
    w = rng.randint(size_min, size_max)
    colors = rng.sample([1, 2, 3, 4, 5, 6, 7, 8], pair_count)
    directions = rng.sample(list(SIDES), pair_count)

    specs: list[dict[str, Any]] = []
    frame_sizes = _frame_size_options(pair_count)
    for color, side in zip(colors, directions):
        fh = rng.choice(frame_sizes)
        fw = rng.choice(frame_sizes)
        inner_h, inner_w = fh - 2, fw - 2
        dims = _choose_key_dims(inner_h, inner_w, side, rng)
        if dims is None:
            return None
        kh, kw = dims
        key_mask = _generate_key_mask(kh, kw, side, rng)
        if key_mask is None:
            return None
        specs.append(
            {
                "color": color,
                "side": side,
                "frame_patch": _make_frame_patch(color, fh, fw, side),
                "key_mask": key_mask,
                "key_patch": _make_key_patch(color, key_mask),
            }
        )

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    objects: list[tuple[str, dict[str, Any], np.ndarray]] = []
    for spec in specs:
        objects.append(("frame", spec, spec["frame_patch"]))
        objects.append(("key", spec, spec["key_patch"]))
    rng.shuffle(objects)

    for kind, spec, patch in objects:
        ph, pw = patch.shape
        placed = False
        for _ in range(300):
            top = rng.randint(0, h - ph)
            left = rng.randint(0, w - pw)
            if not _can_place(blocked, top, left, patch.shape):
                continue
            _stamp(grid, blocked, top, left, patch)
            spec[f"{kind}_pos"] = (top, left)
            placed = True
            break
        if not placed:
            return None

    expected = _spec_output(grid, specs)
    try:
        observed = main(grid)
        parsed = _extract_pairs(grid)
    except AssertionError:
        return None

    if len(parsed) != pair_count:
        return None
    if not np.array_equal(observed, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError("failed to generate a valid input")
