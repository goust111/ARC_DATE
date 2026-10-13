from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _pad_union(masks: list[np.ndarray]) -> np.ndarray:
    h = max(mask.shape[0] for mask in masks)
    w = max(mask.shape[1] for mask in masks)
    out = np.zeros((h, w), dtype=bool)
    for mask in masks:
        mh, mw = mask.shape
        out[:mh, :mw] |= mask
    return out


def _paste_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    rows, cols = np.nonzero(mask)
    grid[top + rows, left + cols] = int(color)


def _marker_columns(grid: np.ndarray, color: int) -> list[int]:
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == int(color))]


def _nearest_green(src_col: int, green_cols: list[int]) -> int:
    return min(green_cols, key=lambda gc: (abs(int(gc) - int(src_col)), -int(gc)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge complementary damaged sprites into one full template, then transfer that template into the paired green lanes.

    Input structure:
    - The background is black.
    - Two full-height red marker columns and two full-height green marker columns define alternating vertical lanes.
    - Orange partial sprite copies appear only in the red lanes, with equal top-edge spacing inside each lane.

    Rule:
    - Crop every orange copy to its own bbox, align the crops at the top-left, pad smaller crops down/right, and take their pixelwise union to recover one complete template.
    - For each red lane, count its orange copies and measure their vertical top-edge spacing; pair that lane with the nearest green marker column, breaking ties toward the right green lane.
    - Recolor every original orange source copy to grey, place the recovered full orange template into the paired green lane from the bottom slot upward using the source lane's spacing, and remove all red/green marker columns to black.

    Key insight:
    - The damaged orange copies carry two independent clues at once: their aligned union reveals what shape to transfer, and their per-lane multiplicity reveals how many copies each green lane must receive.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    red_cols = sorted(_marker_columns(g, RED))
    green_cols = sorted(_marker_columns(g, GREEN))
    assert len(red_cols) == 2 and len(green_cols) == 2

    orange_comps = _components(g == ORANGE)
    assert orange_comps, "expected orange partial sprites in the red lanes"

    lane_parts: dict[int, list[dict[str, object]]] = {col: [] for col in red_cols}
    crops: list[np.ndarray] = []
    for comp in orange_comps:
        r0, c0, r1, c1 = _bbox(comp)
        lane_col = c0 - 1
        assert lane_col in lane_parts, "every orange sprite must start immediately right of a red marker"
        local = _crop_mask(comp)
        lane_parts[lane_col].append(
            {
                "bbox": (r0, c0, r1, c1),
                "mask": local,
            }
        )
        crops.append(local)

    template = _pad_union(crops)
    th, tw = template.shape

    out = g.copy()
    for col in red_cols:
        sprites = lane_parts[col]
        assert len(sprites) >= 2
        tops = sorted(int(part["bbox"][0]) for part in sprites)
        period = tops[1] - tops[0]
        assert period > 0
        if len(tops) >= 3:
            assert tops[2] - tops[1] == period

        for part in sprites:
            r0, c0, _r1, _c1 = part["bbox"]
            mask = part["mask"]
            rows, cols = np.nonzero(mask)
            out[r0 + rows, c0 + cols] = GREY

        dst_col = _nearest_green(col, green_cols)
        left = dst_col + 1
        for idx in range(len(sprites)):
            top = h - th - idx * period
            assert top >= 0
            assert left + tw <= w
            _paste_mask(out, top, left, template, ORANGE)

    for col in red_cols + green_cols:
        out[:, col] = BG
    return out


def _connected_template(mask: np.ndarray) -> bool:
    coords = list(map(tuple, np.argwhere(mask)))
    if not coords:
        return False
    seen = {coords[0]}
    q: deque[tuple[int, int]] = deque([coords[0]])
    coord_set = set(coords)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coord_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(coords)


def _generate_template(rng: random.Random) -> np.ndarray:
    for _ in range(300):
        h = rng.randint(3, 4)
        w = rng.randint(3, 4)
        target = rng.randint(5, min(10, h * w - 1))
        cells = {(0, 0)}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not _connected_template(mask):
            continue
        if not mask[h - 1].any() or not mask[:, w - 1].any():
            continue
        if int(mask.sum()) < 5 or int(mask.sum()) > 10:
            continue
        if int(mask[h - 1].sum()) > 2 or int(mask[:, w - 1].sum()) > 2:
            continue
        if np.all(mask):
            continue
        inner_candidates = [
            (r, c)
            for r, c in np.argwhere(mask)
            if not (r == 0 and c == 0) and r > 0 and c > 0
        ]
        if len(inner_candidates) < 2:
            continue
        return mask
    raise AssertionError("failed to generate a valid template")


def _choose_counts(rng: random.Random) -> tuple[int, int]:
    return rng.choice([(2, 3), (3, 2), (2, 2), (3, 3)])


def _choose_layout(template_w: int, rng: random.Random) -> dict[str, object]:
    layouts = [
        {"widths": [4, 4, 4, 4], "gaps": [1, 1, 1], "branch": "tie_right"},
        {"widths": [4, 5, 4, 4], "gaps": [1, 1, 1], "branch": "right_nearer"},
        {"widths": [4, 4, 4, 4], "gaps": [1, 2, 1], "branch": "right_nearer"},
    ]
    candidates = [layout for layout in layouts if min(layout["widths"]) >= template_w]
    assert candidates
    layout = dict(rng.choice(candidates))
    marker_cols = []
    cur = 0
    for idx, width in enumerate(layout["widths"]):
        marker_cols.append(cur)
        cur += 1 + width
        if idx < 3:
            cur += layout["gaps"][idx]
    layout["marker_cols"] = marker_cols
    layout["width"] = cur
    return layout


def _build_missing_sets(
    template: np.ndarray, counts: tuple[int, int], rng: random.Random
) -> tuple[list[set[tuple[int, int]]], list[set[tuple[int, int]]]]:
    h, w = template.shape
    coords = [tuple(map(int, rc)) for rc in np.argwhere(template)]
    bottom = {(h - 1, c) for c in range(w) if template[h - 1, c]}
    right = {(r, w - 1) for r in range(h) if template[r, w - 1]}
    stable = [cell for cell in coords if cell != (0, 0)]
    common_candidates = [cell for cell in stable if cell[0] > 0 and cell[1] > 0]
    assert len(common_candidates) >= 2

    for _ in range(300):
        common_left, common_right = rng.sample(common_candidates, 2)
        left_sets: list[set[tuple[int, int]]] = [{common_left} | set(bottom)]
        right_sets: list[set[tuple[int, int]]] = [{common_right} | set(right)]

        for lane_sets, common_cell, other_common, needed in (
            (left_sets, common_left, common_right, counts[0]),
            (right_sets, common_right, common_left, counts[1]),
        ):
            while len(lane_sets) < needed:
                miss = {common_cell}
                extra_pool = [
                    cell
                    for cell in coords
                    if cell not in {common_cell, other_common, (0, 0)}
                ]
                rng.shuffle(extra_pool)
                extra_count = rng.randint(0, 2)
                for cell in extra_pool:
                    if len(miss) >= 1 + extra_count:
                        break
                    miss.add(cell)
                lane_sets.append(miss)

        all_sets = left_sets + right_sets
        if not all(1 <= len(miss) <= 3 for miss in all_sets):
            continue

        global_union = np.zeros_like(template, dtype=bool)
        left_union = np.zeros_like(template, dtype=bool)
        right_union = np.zeros_like(template, dtype=bool)
        for idx, miss in enumerate(all_sets):
            partial = template.copy()
            for r, c in miss:
                partial[r, c] = False
            if not partial[0, 0]:
                break
            global_union |= partial
            if idx < counts[0]:
                left_union |= partial
            else:
                right_union |= partial
        else:
            if not np.array_equal(global_union, template):
                continue
            if np.array_equal(left_union, template) or np.array_equal(right_union, template):
                continue
            return left_sets, right_sets

    raise AssertionError("failed to generate complementary missing sets")


def _render_input(meta: dict[str, object]) -> np.ndarray:
    h = int(meta["height"])
    w = int(meta["width"])
    grid = np.zeros((h, w), dtype=int)

    for col in meta["red_cols"]:
        grid[:, int(col)] = RED
    for col in meta["green_cols"]:
        grid[:, int(col)] = GREEN

    template = np.asarray(meta["template"], dtype=bool)
    th, tw = template.shape
    _ = (th, tw)

    for lane in meta["red_lanes"]:
        marker = int(lane["marker"])
        left = marker + 1
        for top, missing in zip(lane["tops"], lane["missing_sets"]):
            partial = template.copy()
            for r, c in missing:
                partial[r, c] = False
            _paste_mask(grid, int(top), left, partial, ORANGE)

    return grid


def _render_expected(meta: dict[str, object]) -> np.ndarray:
    h = int(meta["height"])
    w = int(meta["width"])
    template = np.asarray(meta["template"], dtype=bool)
    th, _tw = template.shape
    out = np.zeros((h, w), dtype=int)

    for lane in meta["red_lanes"]:
        marker = int(lane["marker"])
        left = marker + 1
        for top, missing in zip(lane["tops"], lane["missing_sets"]):
            partial = template.copy()
            for r, c in missing:
                partial[r, c] = False
            _paste_mask(out, int(top), left, partial, GREY)

    for lane in meta["red_lanes"]:
        dst = int(lane["green"])
        left = dst + 1
        for idx in range(int(lane["count"])):
            top = h - th - idx * int(lane["period"])
            _paste_mask(out, top, left, template, ORANGE)

    return out


def _validate_meta(meta: dict[str, object], inp: np.ndarray, out: np.ndarray) -> None:
    template = np.asarray(meta["template"], dtype=bool)
    h, w = inp.shape
    assert h <= 30 and w <= 30
    assert np.array_equal(inp[:, meta["red_cols"][0]], np.full(h, RED))
    assert np.array_equal(inp[:, meta["green_cols"][1]], np.full(h, GREEN))

    all_partials = []
    shrunk_h = False
    shrunk_w = False
    for lane in meta["red_lanes"]:
        lane_union = np.zeros_like(template, dtype=bool)
        for missing in lane["missing_sets"]:
            partial = template.copy()
            for r, c in missing:
                partial[r, c] = False
            assert 1 <= len(missing) <= 3
            assert not np.array_equal(partial, template)
            all_partials.append(partial)
            lane_union |= partial
            if partial.shape[0] > 0:
                box = _bbox(list(map(tuple, np.argwhere(partial))))
                if box[2] - box[0] + 1 < template.shape[0]:
                    shrunk_h = True
                if box[3] - box[1] + 1 < template.shape[1]:
                    shrunk_w = True
        assert not np.array_equal(lane_union, template)
    assert shrunk_h and shrunk_w
    assert np.array_equal(np.logical_or.reduce(all_partials), template)

    right_red = meta["red_cols"][1]
    green_cols = meta["green_cols"]
    dist_left = abs(right_red - green_cols[0])
    dist_right = abs(right_red - green_cols[1])
    if meta["branch"] == "tie_right":
        assert dist_left == dist_right
    else:
        assert dist_right < dist_left

    solved = main(inp)
    assert np.array_equal(solved, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        template = _generate_template(rng)
        th, tw = template.shape

        layout = _choose_layout(tw, rng)
        red_cols = [layout["marker_cols"][0], layout["marker_cols"][2]]
        green_cols = [layout["marker_cols"][1], layout["marker_cols"][3]]

        left_count, right_count = _choose_counts(rng)
        left_period = th + rng.randint(1, 2)
        right_period = th + rng.randint(1, 2)
        min_h = max(
            16,
            th + (left_count - 1) * left_period,
            th + (right_count - 1) * right_period,
        )
        if min_h > 22:
            continue
        height = rng.randint(min_h, 22)

        left_top_max = height - (th + (left_count - 1) * left_period)
        right_top_max = height - (th + (right_count - 1) * right_period)
        left_start = rng.randint(0, left_top_max)
        right_start = rng.randint(0, right_top_max)
        left_tops = [left_start + idx * left_period for idx in range(left_count)]
        right_tops = [right_start + idx * right_period for idx in range(right_count)]

        left_missing, right_missing = _build_missing_sets(template, (left_count, right_count), rng)

        meta = {
            "template": template,
            "height": height,
            "width": int(layout["width"]),
            "red_cols": red_cols,
            "green_cols": green_cols,
            "branch": layout["branch"],
            "red_lanes": [
                {
                    "marker": red_cols[0],
                    "green": green_cols[0],
                    "count": left_count,
                    "period": left_period,
                    "tops": left_tops,
                    "missing_sets": left_missing,
                },
                {
                    "marker": red_cols[1],
                    "green": green_cols[1],
                    "count": right_count,
                    "period": right_period,
                    "tops": right_tops,
                    "missing_sets": right_missing,
                },
            ],
        }

        inp = _render_input(meta)
        out = _render_expected(meta)
        try:
            _validate_meta(meta, inp, out)
        except AssertionError:
            continue
        return inp

    raise AssertionError("generate_input failed after repeated attempts")
