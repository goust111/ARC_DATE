from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS4 = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
SIDES = ("top", "right", "bottom", "left")
PALETTE = list(range(10))
GEN_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _mode_int(values: list[int]) -> int:
    return Counter(map(int, values)).most_common(1)[0][0]


def _outside_coord(slot: dict) -> tuple[int, int]:
    ar, ac = slot["anchor"]
    dr, dc = DIRS4[slot["side"]]
    return ar + dr, ac + dc


def _slot_sequence(rect: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = rect
    slots: list[dict] = []
    order = 0

    for c in range(c0 + 1, c1):
        slots.append({"side": "top", "anchor": (r0, c), "order": order})
        order += 1
    for r in range(r0 + 1, r1):
        slots.append({"side": "right", "anchor": (r, c1), "order": order})
        order += 1
    for c in range(c1 - 1, c0, -1):
        slots.append({"side": "bottom", "anchor": (r1, c), "order": order})
        order += 1
    for r in range(r1 - 1, r0, -1):
        slots.append({"side": "left", "anchor": (r, c0), "order": order})
        order += 1

    return slots


def _infer_rectangle_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    assert int(row_counts.max()) >= 3 and int(col_counts.max()) >= 3

    body_rows = np.flatnonzero(row_counts >= max(3, int(row_counts.max()) - 2))
    body_cols = np.flatnonzero(col_counts >= max(3, int(col_counts.max()) - 2))
    assert body_rows.size >= 3 and body_cols.size >= 3

    lefts = [int(np.flatnonzero(mask[r])[0]) for r in body_rows]
    rights = [int(np.flatnonzero(mask[r])[-1]) for r in body_rows]
    tops = [int(np.flatnonzero(mask[:, c])[0]) for c in body_cols]
    bottoms = [int(np.flatnonzero(mask[:, c])[-1]) for c in body_cols]

    r0 = _mode_int(tops)
    c0 = _mode_int(lefts)
    r1 = _mode_int(bottoms)
    c1 = _mode_int(rights)
    assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4
    return r0, c0, r1, c1


def _feature_type(mask: np.ndarray, slot: dict) -> str | None:
    ar, ac = slot["anchor"]
    orow, ocol = _outside_coord(slot)
    notch = not bool(mask[ar, ac])
    tab = in_bounds(mask.shape, orow, ocol) and bool(mask[orow, ocol])
    assert not (notch and tab)
    if notch:
        return "notch"
    if tab:
        return "tab"
    return None


def _collect_features(mask: np.ndarray, rect: tuple[int, int, int, int]) -> list[dict]:
    features: list[dict] = []
    for slot in _slot_sequence(rect):
        kind = _feature_type(mask, slot)
        if kind is not None:
            features.append(
                {
                    "side": slot["side"],
                    "anchor": slot["anchor"],
                    "outside": _outside_coord(slot),
                    "order": slot["order"],
                    "type": kind,
                }
            )
    assert features
    return features


def _render_grid(
    shape: tuple[int, int],
    bg: int,
    fg: int,
    rect: tuple[int, int, int, int],
    features: list[dict],
    types: list[str],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    r0, c0, r1, c1 = rect
    out[r0 : r1 + 1, c0 : c1 + 1] = int(fg)

    for feature, kind in zip(features, types):
        ar, ac = feature["anchor"]
        orow, ocol = _outside_coord(feature)
        if kind == "notch":
            out[ar, ac] = int(bg)
        else:
            out[ar, ac] = int(fg)
            if in_bounds(shape, orow, ocol):
                out[orow, ocol] = int(fg)
    return out


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = get_bg(g)
    fg_colors = colors_present(g, exclude={bg})
    assert len(fg_colors) == 1
    fg = int(fg_colors[0])
    mask = g == fg
    rect = _infer_rectangle_bbox(mask)
    features = _collect_features(mask, rect)
    return {"grid": g, "bg": bg, "fg": fg, "rect": rect, "features": features}


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Rotate boundary tab/notch types one step clockwise around the same rectangle slots.

    Input structure:
    - The grid contains a dominant background and one large solid rectangle in a single foreground color.
    - Some non-corner perimeter slots carry a one-pixel tab just outside the rectangle, while others carry a one-pixel notch cut into the edge.
    - Feature slots are separated from each other, and the rectangle body is otherwise solid.

    Rule:
    - Recover the underlying clean rectangle and list all feature slots in clockwise perimeter order.
    - Keep every feature slot in the same place, but replace its type with the type of the previous slot in that clockwise list.
    - Leave the rest of the rectangle unchanged.

    Key insight:
    - The perimeter positions are fixed carriers; only the tab-versus-notch attribute cycles along the boundary.
    """
    info = _analyze_grid(np.asarray(input_grid, dtype=int))
    types = [feature["type"] for feature in info["features"]]
    shifted = [types[-1]] + types[:-1]
    return _render_grid(
        info["grid"].shape,
        info["bg"],
        info["fg"],
        info["rect"],
        info["features"],
        shifted,
    )


def _anchor_far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1


def _choose_feature_slots(
    rng: random.Random, rect: tuple[int, int, int, int]
) -> list[dict] | None:
    slots = _slot_sequence(rect)
    by_side = {side: [slot for slot in slots if slot["side"] == side] for side in SIDES}
    for _ in range(120):
        target_side_count = 4 if rng.random() < 0.55 else 3
        active_sides = rng.sample(list(SIDES), target_side_count)
        selected: list[dict] = []

        for side in active_sides:
            candidates = by_side[side][:]
            rng.shuffle(candidates)
            placed = False
            for slot in candidates:
                if all(_anchor_far_enough(slot["anchor"], other["anchor"]) for other in selected):
                    selected.append(slot)
                    placed = True
                    break
            if not placed:
                break
        if len(selected) != target_side_count:
            continue

        target_count = rng.randint(max(4, target_side_count), min(8, len(slots)) + 1)
        remaining = [slot for slot in slots if slot not in selected]
        rng.shuffle(remaining)
        for slot in remaining:
            if len(selected) >= target_count:
                break
            if all(_anchor_far_enough(slot["anchor"], other["anchor"]) for other in selected):
                selected.append(slot)

        if len(selected) < 4:
            continue
        if len({slot["side"] for slot in selected}) < target_side_count:
            continue
        selected.sort(key=lambda slot: slot["order"])
        return selected
    return None


def _choose_types(rng: random.Random, n_features: int) -> list[str] | None:
    for _ in range(120):
        tab_count = rng.randint(1, n_features - 1)
        types = ["tab"] * tab_count + ["notch"] * (n_features - tab_count)
        rng.shuffle(types)
        if types != [types[-1]] + types[:-1]:
            return types
    return None


def _pick_geometry(
    rng: random.Random,
) -> tuple[int, int, int, int, int, int] | None:
    for _ in range(120):
        height = rng.randint(12, 28)
        width = rng.randint(12, 28)
        rect_h = rng.randint(5, min(16, height - 2))
        rect_w = rng.randint(5, min(16, width - 2))
        if rect_h * rect_w >= height * width * 0.45:
            continue
        r0 = rng.randint(1, height - rect_h - 1)
        c0 = rng.randint(1, width - rect_w - 1)
        return height, width, r0, c0, rect_h, rect_w
    return None


def _build_candidate() -> np.ndarray | None:
    rng = random.Random()
    colors = rng.sample(PALETTE, 2)
    bg, fg = colors[0], colors[1]

    geometry = _pick_geometry(rng)
    if geometry is None:
        return None
    height, width, r0, c0, rect_h, rect_w = geometry
    rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

    features = _choose_feature_slots(rng, rect)
    if features is None:
        return None
    types = _choose_types(rng, len(features))
    if types is None:
        return None

    grid = _render_grid((height, width), bg, fg, rect, features, types)
    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def _validate_generated(grid: np.ndarray) -> None:
    info = _analyze_grid(grid)
    features = info["features"]
    types = [feature["type"] for feature in features]

    assert len(features) >= 4
    assert len({feature["side"] for feature in features}) >= 3
    assert {"tab", "notch"} <= set(types)

    out = main(grid)
    out_info = _analyze_grid(out)
    out_types = [feature["type"] for feature in out_info["features"]]
    expected = [types[-1]] + types[:-1]

    assert info["rect"] == out_info["rect"]
    assert [feature["anchor"] for feature in features] == [feature["anchor"] for feature in out_info["features"]]
    assert out_types == expected
    assert np.array_equal(
        out,
        _render_grid(out.shape, info["bg"], info["fg"], info["rect"], features, expected),
    )
    assert not np.array_equal(grid, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
