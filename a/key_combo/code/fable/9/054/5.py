import random
from collections import Counter, deque

import numpy as np


PALETTE = list(range(1, 10))
DIR_STEP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _max_marker_offset(length: int) -> int:
    return (int(length) - 3) // 2


def _marker_position(rect: dict, direction: str, offset: int | None = None) -> tuple[int, int]:
    r0, c0 = int(rect["r0"]), int(rect["c0"])
    h, w = int(rect["h"]), int(rect["w"])
    cr = r0 + h // 2
    cc = c0 + w // 2
    if direction in {"left", "right"}:
        max_offset = _max_marker_offset(w)
    else:
        max_offset = _max_marker_offset(h)
    if max_offset < 1:
        raise AssertionError("rectangle too small for directional marker")
    d = int(offset if offset is not None else rect.get("marker_offset", 1))
    if not (1 <= d <= max_offset):
        raise AssertionError("invalid marker offset")
    dr, dc = DIR_STEP[direction]
    return cr + dr * d, cc + dc * d


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    rects: list[dict] = []
    for comp in _components(g != bg):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == bg):
            raise AssertionError("foreground component must fill its bounding box")
        colors, counts = np.unique(patch, return_counts=True)
        if len(colors) != 2:
            raise AssertionError("each rectangle must have exactly one marker color")
        body = int(colors[int(np.argmax(counts))])
        marker_candidates = [(int(col), int(cnt)) for col, cnt in zip(colors, counts) if int(col) != body]
        if len(marker_candidates) != 1 or marker_candidates[0][1] != 1:
            raise AssertionError("marker must be a single pixel of a different color")
        marker_color = marker_candidates[0][0]
        h, w = patch.shape
        if h % 2 == 0 or w % 2 == 0:
            raise AssertionError("generated rectangles use odd sizes for a unique center cell")
        marker_r, marker_c = map(int, np.argwhere(patch == marker_color)[0])
        marker_r += r0
        marker_c += c0
        center_r = r0 + h // 2
        center_c = c0 + w // 2
        dr = marker_r - center_r
        dc = marker_c - center_c
        if (dr == 0) == (dc == 0):
            raise AssertionError("marker offset must be nonzero and axis-aligned")
        if dr == 0:
            direction = "right" if dc > 0 else "left"
            if marker_c in {c0, c1}:
                raise AssertionError("marker must be strictly inside the rectangle")
            marker_offset = abs(dc)
        else:
            direction = "down" if dr > 0 else "up"
            if marker_r in {r0, r1}:
                raise AssertionError("marker must be strictly inside the rectangle")
            marker_offset = abs(dr)
        rects.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "h": h,
                "w": w,
                "body": body,
                "marker_color": marker_color,
                "marker_r": marker_r,
                "marker_c": marker_c,
                "direction": direction,
                "marker_offset": int(marker_offset),
            }
        )
    rects.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["body"]))
    return rects


def _gap_in_direction(src: dict, dst: dict, direction: str) -> int | None:
    if direction == "right":
        if dst["c0"] > src["c1"] and _interval_overlap(src["r0"], src["r1"], dst["r0"], dst["r1"]):
            return int(dst["c0"] - src["c1"] - 1)
    elif direction == "left":
        if dst["c1"] < src["c0"] and _interval_overlap(src["r0"], src["r1"], dst["r0"], dst["r1"]):
            return int(src["c0"] - dst["c1"] - 1)
    elif direction == "down":
        if dst["r0"] > src["r1"] and _interval_overlap(src["c0"], src["c1"], dst["c0"], dst["c1"]):
            return int(dst["r0"] - src["r1"] - 1)
    else:
        if dst["r1"] < src["r0"] and _interval_overlap(src["c0"], src["c1"], dst["c0"], dst["c1"]):
            return int(src["r0"] - dst["r1"] - 1)
    return None


def _signed_axis_gap(src: dict, dst: dict, direction: str) -> int | None:
    if direction == "right" and dst["c0"] > src["c1"]:
        return int(dst["c0"] - src["c1"] - 1)
    if direction == "left" and dst["c1"] < src["c0"]:
        return int(src["c0"] - dst["c1"] - 1)
    if direction == "down" and dst["r0"] > src["r1"]:
        return int(dst["r0"] - src["r1"] - 1)
    if direction == "up" and dst["r1"] < src["r0"]:
        return int(src["r0"] - dst["r1"] - 1)
    return None


def _nearest_rect(rect: dict, rects: list[dict]) -> dict | None:
    candidates: list[tuple[int, int, dict]] = []
    for idx, other in enumerate(rects):
        if other is rect:
            continue
        gap = _gap_in_direction(rect, other, rect["direction"])
        if gap is not None:
            candidates.append((gap, idx, other))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) >= 2 and candidates[0][0] == candidates[1][0]:
        raise AssertionError("nearest rectangle must be unique")
    return candidates[0][2]


def _final_bbox(rect: dict, target: dict | None) -> tuple[int, int, int, int]:
    r0, c0 = int(rect["r0"]), int(rect["c0"])
    h, w = int(rect["h"]), int(rect["w"])
    if target is None or int(target["body"]) != int(rect["marker_color"]):
        return r0, c0, r0 + h - 1, c0 + w - 1
    direction = rect["direction"]
    if direction == "right":
        c0 = int(target["c0"]) - w
    elif direction == "left":
        c0 = int(target["c1"]) + 1
    elif direction == "down":
        r0 = int(target["r0"]) - h
    else:
        r0 = int(target["r1"]) + 1
    return r0, c0, r0 + h - 1, c0 + w - 1


def _render_final(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["final_bbox"]
        if r0 < 0 or c0 < 0 or r1 >= shape[0] or c1 >= shape[1]:
            raise AssertionError("translated rectangle left the canvas")
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(region):
            raise AssertionError("final rectangles overlap")
        region[:, :] = True
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rect["body"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked solid rectangle toward its permitted docking partner, then erase the markers.

    Input structure:
    - The background is the most common color.
    - Each foreground component is a solid monochrome rectangle with one strictly
      interior marker pixel of a different color.
    - The marker sits on the rectangle's horizontal or vertical center line, away
      from the center cell, so its offset encodes one cardinal direction.

    Rule:
    - For each rectangle, read the marker's offset direction.
    - Find the nearest other rectangle in that direction whose orthogonal span
      overlaps, so a face-to-face docking slide is possible.
    - If that nearest rectangle's body color matches the marker color, translate
      the rectangle along the encoded direction until the two rectangles become
      adjacent; otherwise it stays put.
    - Finally, repaint every marker cell with its rectangle's own body color.

    Key insight:
    - The marker has two simultaneous roles: its position selects the slide
      direction, and its color grants or denies the docking permission.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    bg = _background_color(g)
    rects = _extract_rectangles(g)
    for rect in rects:
        rect["target"] = _nearest_rect(rect, rects)
        rect["final_bbox"] = _final_bbox(rect, rect["target"])
    return _render_final(g.shape, rects)


def _weighted_choice(options: list[int]) -> int:
    return int(random.choice(options))


def _pick_main_size(axis: str) -> tuple[int, int]:
    if axis == "horizontal":
        return _weighted_choice([3, 5, 5]), _weighted_choice([5, 5, 7])
    return _weighted_choice([5, 5, 7]), _weighted_choice([3, 5, 5])


def _pick_decoy_size(axis: str) -> tuple[int, int]:
    if axis == "horizontal":
        return _weighted_choice([5, 5, 7]), _weighted_choice([3, 5])
    return _weighted_choice([3, 5]), _weighted_choice([5, 5, 7])


def _assign_colors(rects: list[dict]) -> None:
    bodies = random.sample(PALETTE, len(rects))
    by_name = {rect["name"]: rect for rect in rects}
    for rect, body in zip(rects, bodies):
        rect["body"] = int(body)
    for rect in rects:
        if rect["role"] == "mover" and rect["mode"] == "active":
            rect["marker_color"] = int(by_name[rect["target_name"]]["body"])
        elif rect["role"] == "mover" and rect["mode"] == "blocked":
            forbidden = {int(rect["body"]), int(by_name[rect["target_name"]]["body"])}
            rect["marker_color"] = int(random.choice([c for c in PALETTE if c not in forbidden]))
        else:
            rect["marker_color"] = int(random.choice([c for c in PALETTE if c != int(rect["body"])]))


def _make_rect(
    name: str,
    role: str,
    mode: str,
    top: int,
    left: int,
    h: int,
    w: int,
    direction: str,
    target_name: str | None,
    false_front_name: str | None = None,
) -> dict:
    axis_len = w if direction in {"left", "right"} else h
    return {
        "name": name,
        "role": role,
        "mode": mode,
        "r0": int(top),
        "c0": int(left),
        "h": int(h),
        "w": int(w),
        "r1": int(top + h - 1),
        "c1": int(left + w - 1),
        "direction": direction,
        "target_name": target_name,
        "false_front_name": false_front_name,
        "marker_offset": random.randint(1, _max_marker_offset(axis_len)),
    }


def _build_horizontal_case(prefix: str, mode: str, include_decoy: bool) -> tuple[list[dict], int, int]:
    direction = random.choice(["left", "right"])
    mover_h, mover_w = _pick_main_size("horizontal")
    anchor_h, anchor_w = _pick_main_size("horizontal")
    decoy_h = decoy_w = 0
    if include_decoy:
        decoy_h, decoy_w = _pick_decoy_size("horizontal")
        gap_min = decoy_w + 2
    else:
        gap_min = 2
    gap = random.randint(gap_min, gap_min + 2)
    main_band_h = max(mover_h, anchor_h)
    mover_top = (main_band_h - mover_h) // 2
    anchor_top = (main_band_h - anchor_h) // 2

    rects: list[dict] = []
    mover_name = f"{prefix}_mover"
    anchor_name = f"{prefix}_anchor"
    decoy_name = f"{prefix}_decoy" if include_decoy else None

    if direction == "right":
        mover_left = 0
        anchor_left = mover_w + gap
        false_name = decoy_name
        mover = _make_rect(
            mover_name,
            "mover",
            mode,
            mover_top,
            mover_left,
            mover_h,
            mover_w,
            "right",
            anchor_name,
            false_name,
        )
        anchor = _make_rect(
            anchor_name,
            "anchor",
            "idle",
            anchor_top,
            anchor_left,
            anchor_h,
            anchor_w,
            "right",
            None,
        )
        rects.extend([mover, anchor])
        patch_w = anchor_left + anchor_w
        if include_decoy:
            decoy_left = random.randint(mover_w + 1, anchor_left - decoy_w - 1)
            decoy_top = main_band_h + 1
            decoy = _make_rect(decoy_name, "decoy", "idle", decoy_top, decoy_left, decoy_h, decoy_w, "down", None)
            rects.append(decoy)
            patch_h = decoy_top + decoy_h
        else:
            patch_h = main_band_h
    else:
        anchor_left = 0
        mover_left = anchor_w + gap
        false_name = decoy_name
        anchor = _make_rect(
            anchor_name,
            "anchor",
            "idle",
            anchor_top,
            anchor_left,
            anchor_h,
            anchor_w,
            "left",
            None,
        )
        mover = _make_rect(
            mover_name,
            "mover",
            mode,
            mover_top,
            mover_left,
            mover_h,
            mover_w,
            "left",
            anchor_name,
            false_name,
        )
        rects.extend([anchor, mover])
        patch_w = mover_left + mover_w
        if include_decoy:
            decoy_left = random.randint(anchor_w + 1, mover_left - decoy_w - 1)
            decoy_top = main_band_h + 1
            decoy = _make_rect(decoy_name, "decoy", "idle", decoy_top, decoy_left, decoy_h, decoy_w, "down", None)
            rects.append(decoy)
            patch_h = decoy_top + decoy_h
        else:
            patch_h = main_band_h
    return rects, int(patch_h), int(patch_w)


def _build_vertical_case(prefix: str, mode: str, include_decoy: bool) -> tuple[list[dict], int, int]:
    direction = random.choice(["up", "down"])
    mover_h, mover_w = _pick_main_size("vertical")
    anchor_h, anchor_w = _pick_main_size("vertical")
    decoy_h = decoy_w = 0
    if include_decoy:
        decoy_h, decoy_w = _pick_decoy_size("vertical")
        gap_min = decoy_h + 2
    else:
        gap_min = 2
    gap = random.randint(gap_min, gap_min + 2)
    main_band_w = max(mover_w, anchor_w)
    mover_left = (main_band_w - mover_w) // 2
    anchor_left = (main_band_w - anchor_w) // 2

    rects: list[dict] = []
    mover_name = f"{prefix}_mover"
    anchor_name = f"{prefix}_anchor"
    decoy_name = f"{prefix}_decoy" if include_decoy else None

    if direction == "down":
        mover_top = 0
        anchor_top = mover_h + gap
        mover = _make_rect(
            mover_name,
            "mover",
            mode,
            mover_top,
            mover_left,
            mover_h,
            mover_w,
            "down",
            anchor_name,
            decoy_name,
        )
        anchor = _make_rect(
            anchor_name,
            "anchor",
            "idle",
            anchor_top,
            anchor_left,
            anchor_h,
            anchor_w,
            "down",
            None,
        )
        rects.extend([mover, anchor])
        patch_h = anchor_top + anchor_h
        if include_decoy:
            decoy_top = random.randint(mover_h + 1, anchor_top - decoy_h - 1)
            decoy_left = main_band_w + 1
            decoy = _make_rect(decoy_name, "decoy", "idle", decoy_top, decoy_left, decoy_h, decoy_w, "right", None)
            rects.append(decoy)
            patch_w = decoy_left + decoy_w
        else:
            patch_w = main_band_w
    else:
        anchor_top = 0
        mover_top = anchor_h + gap
        anchor = _make_rect(
            anchor_name,
            "anchor",
            "idle",
            anchor_top,
            anchor_left,
            anchor_h,
            anchor_w,
            "up",
            None,
        )
        mover = _make_rect(
            mover_name,
            "mover",
            mode,
            mover_top,
            mover_left,
            mover_h,
            mover_w,
            "up",
            anchor_name,
            decoy_name,
        )
        rects.extend([anchor, mover])
        patch_h = mover_top + mover_h
        if include_decoy:
            decoy_top = random.randint(anchor_h + 1, mover_top - decoy_h - 1)
            decoy_left = main_band_w + 1
            decoy = _make_rect(decoy_name, "decoy", "idle", decoy_top, decoy_left, decoy_h, decoy_w, "right", None)
            rects.append(decoy)
            patch_w = decoy_left + decoy_w
        else:
            patch_w = main_band_w
    return rects, int(patch_h), int(patch_w)


def _shift_rects(rects: list[dict], dr: int, dc: int) -> list[dict]:
    shifted: list[dict] = []
    for rect in rects:
        new_rect = dict(rect)
        new_rect["r0"] += int(dr)
        new_rect["r1"] += int(dr)
        new_rect["c0"] += int(dc)
        new_rect["c1"] += int(dc)
        shifted.append(new_rect)
    return shifted


def _render_input(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        if np.any(occupied[r0 : r1 + 1, c0 : c1 + 1]):
            raise AssertionError("input rectangles overlap")
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect["body"])
    for rect in rects:
        mr, mc = _marker_position(rect, rect["direction"])
        rect["marker_r"], rect["marker_c"] = int(mr), int(mc)
        grid[mr, mc] = int(rect["marker_color"])
    return grid


def _oracle_output(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    by_name = {rect["name"]: rect for rect in rects}
    expected_rects: list[dict] = []
    for rect in rects:
        target = by_name.get(rect["target_name"]) if rect["target_name"] else None
        new_rect = dict(rect)
        new_rect["final_bbox"] = _final_bbox(rect, target)
        expected_rects.append(new_rect)
    return _render_final(shape, expected_rects)


def _match_by_bbox(meta_rects: list[dict], parsed_rects: list[dict]) -> dict[str, dict]:
    lookup = {
        (rect["r0"], rect["c0"], rect["r1"], rect["c1"], rect["body"]): rect for rect in parsed_rects
    }
    out: dict[str, dict] = {}
    for rect in meta_rects:
        key = (rect["r0"], rect["c0"], rect["r1"], rect["c1"], rect["body"])
        if key not in lookup:
            raise AssertionError("failed to recover generated rectangle from the rendered grid")
        out[rect["name"]] = lookup[key]
    return out


def _validate_scene(shape: tuple[int, int], rects: list[dict], grid: np.ndarray) -> None:
    if len([rect for rect in rects if rect["role"] == "mover" and rect["mode"] == "active"]) != 1:
        raise AssertionError("scene must contain exactly one permitted mover")
    if len([rect for rect in rects if rect["role"] == "mover" and rect["mode"] == "blocked"]) != 1:
        raise AssertionError("scene must contain exactly one blocked mover")
    if len([rect for rect in rects if rect["mode"] == "idle"]) < 2:
        raise AssertionError("scene must contain idle rectangles")

    counts = np.bincount(grid.ravel(), minlength=10)
    if counts[0] <= int(counts[1:].max()):
        raise AssertionError("background must remain the most common color")

    parsed = _extract_rectangles(grid)
    if len(parsed) != len(rects):
        raise AssertionError("rectangle count changed after rendering")
    parsed_by_name = _match_by_bbox(rects, parsed)

    for rect in rects:
        parsed_rect = parsed_by_name[rect["name"]]
        if parsed_rect["direction"] != rect["direction"]:
            raise AssertionError("marker direction changed after parsing")
        if parsed_rect["marker_color"] != rect["marker_color"]:
            raise AssertionError("marker color changed after parsing")
        if parsed_rect["marker_offset"] != rect["marker_offset"]:
            raise AssertionError("marker offset changed after parsing")

    parsed_rects = list(parsed_by_name.values())
    for rect in rects:
        parsed_rect = parsed_by_name[rect["name"]]
        target = _nearest_rect(parsed_rect, parsed_rects)
        expected_target_name = rect["target_name"]
        if expected_target_name is None:
            if target is not None:
                raise AssertionError("idle rectangle unexpectedly found a target")
        else:
            expected_target = parsed_by_name[expected_target_name]
            if target is not expected_target:
                raise AssertionError("nearest directional target is wrong")

    for mover in [rect for rect in rects if rect["false_front_name"]]:
        decoy = next(rect for rect in rects if rect["name"] == mover["false_front_name"])
        anchor = next(rect for rect in rects if rect["name"] == mover["target_name"])
        decoy_gap = _signed_axis_gap(mover, decoy, mover["direction"])
        anchor_gap = _signed_axis_gap(mover, anchor, mover["direction"])
        if decoy_gap is None or anchor_gap is None:
            raise AssertionError("false-front decoy is not in the signed half-plane")
        if mover["direction"] in {"left", "right"}:
            if _interval_overlap(mover["r0"], mover["r1"], decoy["r0"], decoy["r1"]):
                raise AssertionError("false-front decoy must not be row-aligned")
        else:
            if _interval_overlap(mover["c0"], mover["c1"], decoy["c0"], decoy["c1"]):
                raise AssertionError("false-front decoy must not be column-aligned")
        if decoy_gap >= anchor_gap:
            raise AssertionError("false-front decoy should be closer than the true target")

    expected = _oracle_output(shape, rects)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        raise AssertionError("solver disagrees with scene oracle")
    if np.array_equal(grid, expected):
        raise AssertionError("scene is degenerate: output equals input")

    by_name = {rect["name"]: rect for rect in rects}
    active = next(rect for rect in rects if rect["role"] == "mover" and rect["mode"] == "active")
    blocked = next(rect for rect in rects if rect["role"] == "mover" and rect["mode"] == "blocked")

    color_flip = grid.copy()
    color_flip[active["marker_r"], active["marker_c"]] = int(active["body"])
    color_flip[active["marker_r"], active["marker_c"]] = int(blocked["body"])
    if np.array_equal(main(color_flip), expected):
        raise AssertionError("active mover should depend on marker color permission")

    dir_flip = grid.copy()
    dir_flip[active["marker_r"], active["marker_c"]] = int(active["body"])
    opposite_pos = _marker_position(active, OPPOSITE[active["direction"]], active["marker_offset"])
    dir_flip[opposite_pos] = int(active["marker_color"])
    if np.array_equal(main(dir_flip), expected):
        raise AssertionError("active mover should depend on marker direction")

    unblock = grid.copy()
    unblock[blocked["marker_r"], blocked["marker_c"]] = int(by_name[blocked["target_name"]]["body"])
    if np.array_equal(main(unblock), expected):
        raise AssertionError("blocked mover should change when its marker color grants docking")


def _generate_scene() -> np.ndarray | None:
    if random.random() < 0.5:
        h_mode, v_mode = "active", "blocked"
    else:
        h_mode, v_mode = "blocked", "active"

    h_rects, hh, hw = _build_horizontal_case("h", h_mode, include_decoy=True)
    v_rects, vh, vw = _build_vertical_case("v", v_mode, include_decoy=random.random() < 0.6)

    min_h = hh + vh + 3
    min_w = hw + vw + 3
    if min_h > 30 or min_w > 30:
        return None

    h_total = random.randint(min_h, min(30, min_h + 4))
    w_total = random.randint(min_w, min(30, min_w + 4))
    h_top = random.randint(1, 2)
    h_left = random.randint(1, 2)
    v_top_min = h_top + hh + 1
    v_left_min = h_left + hw + 1
    v_top_max = h_total - vh - 1
    v_left_max = w_total - vw - 1
    if v_top_min > v_top_max or v_left_min > v_left_max:
        return None

    v_top = random.randint(v_top_min, v_top_max)
    v_left = random.randint(v_left_min, v_left_max)
    scene_rects = _shift_rects(h_rects, h_top, h_left) + _shift_rects(v_rects, v_top, v_left)
    _assign_colors(scene_rects)
    grid = _render_input((h_total, w_total), scene_rects)
    _validate_scene((h_total, w_total), scene_rects, grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid docking scene")
