from __future__ import annotations

import random
from typing import Any

import numpy as np


BG = 0
MARKER = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
GEN_RETRIES = 800


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _is_container_frame(grid: np.ndarray, coords: np.ndarray, color: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(coords) != expected:
        return False
    pts = {tuple(map(int, p)) for p in coords}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in pts):
                return False
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return False
    if int((inner == MARKER).sum()) != 1:
        return False
    return bool(np.all((inner == BG) | (inner == MARKER)))


def _parse_scene(grid: np.ndarray) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in (BG, MARKER)]
    containers: dict[int, dict[str, Any]] = {}
    raw_fragments: dict[int, list[np.ndarray]] = {}

    for color in colors:
        frame: dict[str, Any] | None = None
        others: list[np.ndarray] = []
        for coords in _components(grid == color):
            if _is_container_frame(grid, coords, color):
                if frame is not None:
                    raise ValueError(f"multiple containers found for color {color}")
                r0, c0, r1, c1 = _bbox(coords)
                marker_local = np.argwhere(grid[r0 + 1 : r1, c0 + 1 : c1] == MARKER)
                mr, mc = map(int, marker_local[0])
                frame = {
                    "bbox": (r0, c0, r1, c1),
                    "marker": (r0 + 1 + mr, c0 + 1 + mc),
                    "coords": coords,
                }
            else:
                others.append(coords)
        if frame is not None:
            containers[color] = frame
            raw_fragments[color] = others

    fragments_by_color: dict[int, list[dict[str, Any]]] = {}
    for color, comps in raw_fragments.items():
        pieces: list[dict[str, Any]] = []
        for coords in comps:
            r0, c0, r1, c1 = _bbox(coords)
            rel = coords - np.array([[r0, c0]], dtype=int)
            pieces.append(
                {
                    "coords": coords,
                    "rel": rel,
                    "bbox": (r0, c0, r1, c1),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                    "area": int(len(coords)),
                }
            )
        fragments_by_color[color] = pieces

    return containers, fragments_by_color


def _place_piece(
    out: np.ndarray,
    top: int,
    left: int,
    color: int,
    piece: dict[str, Any],
) -> None:
    for rr, cc in piece["rel"]:
        out[top + int(rr), left + int(cc)] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each color's scattered fragments into the matching hollow container.

    Containers are 1-pixel frames with a single maroon marker somewhere inside.
    The marker is only a distractor; the actual placement rule is:
    sort same-color fragments by area descending, then shelf-pack them inside the
    container from the inner top-left corner, left-to-right with row wraps.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    containers, fragments_by_color = _parse_scene(g)

    for pieces in fragments_by_color.values():
        for piece in pieces:
            coords = piece["coords"]
            out[coords[:, 0], coords[:, 1]] = BG

    for color, container in containers.items():
        r0, c0, r1, c1 = container["bbox"]
        inner_top = r0 + 1
        inner_left = c0 + 1
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        out[inner_top:r1, inner_left:c1] = BG

        pieces = list(fragments_by_color.get(color, []))
        pieces.sort(
            key=lambda piece: (
                -int(piece["area"]),
                int(piece["h"]) * int(piece["w"]),
                int(piece["h"]),
                int(piece["w"]),
                int(piece["bbox"][0]),
                int(piece["bbox"][1]),
            )
        )

        cur_r = 0
        cur_c = 0
        row_h = 0
        for piece in pieces:
            ph = int(piece["h"])
            pw = int(piece["w"])
            if cur_c > 0 and cur_c + pw > inner_w:
                cur_r += row_h
                cur_c = 0
                row_h = 0
            if pw > inner_w or cur_r + ph > inner_h:
                raise ValueError("fragment layout does not fit container interior")
            _place_piece(out, inner_top + cur_r, inner_left + cur_c, color, piece)
            cur_c += pw
            row_h = max(row_h, ph)

    return out


def _partition_total(rng: random.Random, total: int, parts: int) -> list[int] | None:
    if parts <= 0 or total < parts:
        return None
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts:
        values.append(cut - prev)
        prev = cut
    values.append(total - prev)
    return values


def _max_distinct_parts(total: int) -> int:
    k = 0
    while (k + 1) * (k + 2) // 2 <= total:
        k += 1
    return k


def _distinct_partition_total(rng: random.Random, total: int, parts: int) -> list[int] | None:
    if parts <= 0 or parts > _max_distinct_parts(total):
        return None
    for _ in range(100):
        values = _partition_total(rng, total, parts)
        if values is None:
            continue
        if len(set(values)) != len(values):
            continue
        return sorted(values, reverse=True)
    return None


def _make_rectangle_piece(h: int, w: int) -> dict[str, Any]:
    rel = np.array([(r, c) for r in range(h) for c in range(w)], dtype=int)
    return {"rel": rel, "h": h, "w": w, "area": h * w}


def _build_fragment_plan(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    wrap: bool,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]], np.ndarray] | None:
    max_rows = min(3, inner_h)
    if wrap and max_rows < 2:
        return None

    for _ in range(200):
        row_count = rng.randint(2, max_rows) if wrap else 1

        remaining_h = inner_h
        row_heights: list[int] = []
        prev_h = 3
        for i in range(row_count):
            rows_left = row_count - i - 1
            max_h = min(3, prev_h, remaining_h - rows_left)
            if max_h < 1:
                row_heights = []
                break
            row_heights.append(rng.randint(1, max_h))
            remaining_h -= row_heights[-1]
            prev_h = row_heights[-1]
        if not row_heights:
            continue

        piece_counts: list[int] = []
        for i in range(row_count):
            if not wrap and i == 0:
                lo = 2
            else:
                lo = 1
            hi = min(3, _max_distinct_parts(inner_w))
            if lo > hi:
                piece_counts = []
                break
            piece_counts.append(rng.randint(lo, hi))
        if not piece_counts:
            continue

        if not (2 <= sum(piece_counts) <= 6):
            continue

        row_widths: list[list[int]] = []
        ok = True
        for i, count in enumerate(piece_counts):
            exact = wrap and i < row_count - 1
            if exact:
                total = inner_w
            else:
                low = max(count * (count + 1) // 2, max(2, inner_w // 2))
                total = rng.randint(low, inner_w)
            widths = _distinct_partition_total(rng, total, count)
            if widths is None:
                ok = False
                break
            row_widths.append(widths)
        if not ok:
            continue

        row_areas = [[row_heights[i] * w for w in widths] for i, widths in enumerate(row_widths)]
        areas = [row_heights[i] * w for i, widths in enumerate(row_widths) for w in widths]
        if len(set(areas)) != len(areas):
            continue
        if areas != sorted(areas, reverse=True):
            continue
        if any(
            row_areas[i][-1] <= row_areas[i + 1][0]
            for i in range(len(row_areas) - 1)
        ):
            continue

        pieces: list[dict[str, Any]] = []
        anchors: list[tuple[int, int]] = []
        filled = np.zeros((inner_h, inner_w), dtype=bool)
        cur_y = 0
        for row_h, widths in zip(row_heights, row_widths):
            cur_x = 0
            for w in widths:
                pieces.append(_make_rectangle_piece(row_h, w))
                anchors.append((cur_y, cur_x))
                filled[cur_y : cur_y + row_h, cur_x : cur_x + w] = True
                cur_x += w
            cur_y += row_h

        return pieces, anchors, filled

    return None


def _choose_marker(
    rng: random.Random,
    filled: np.ndarray,
) -> tuple[int, int]:
    h, w = filled.shape
    max_r = max(1, h - 1)
    max_c = max(1, w - 1)
    candidates = [(r, c) for r in range(max_r) for c in range(max_c)]
    upper_left = [
        (r, c)
        for r, c in candidates
        if r < max(1, (h + 1) // 2) and c < max(1, (w + 1) // 2)
    ]
    if upper_left:
        candidates = upper_left

    want_covered = rng.random() < 0.5
    filtered = [(r, c) for r, c in candidates if bool(filled[r, c]) == want_covered]
    if filtered:
        return rng.choice(filtered)
    return rng.choice(candidates)


def _find_free_spot(
    rng: random.Random,
    occupied: np.ndarray,
    h: int,
    w: int,
    pad: int = 1,
    tries: int = 400,
) -> tuple[int, int] | None:
    grid_h, grid_w = occupied.shape
    if h + 2 * pad > grid_h or w + 2 * pad > grid_w:
        return None
    for _ in range(tries):
        r = rng.randint(pad, grid_h - h - pad)
        c = rng.randint(pad, grid_w - w - pad)
        rr0 = max(0, r - pad)
        rr1 = min(grid_h, r + h + pad)
        cc0 = max(0, c - pad)
        cc1 = min(grid_w, c + w + pad)
        if occupied[rr0:rr1, cc0:cc1].any():
            continue
        return r, c
    return None


def _paint_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    color: int,
) -> None:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color


def _paint_piece(grid: np.ndarray, top: int, left: int, color: int, piece: dict[str, Any]) -> None:
    for rr, cc in piece["rel"]:
        grid[top + int(rr), left + int(cc)] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(22, 34)
    grid_w = rng.randint(22, 34)
    num_containers = rng.randint(2, 4)
    colors = rng.sample(PALETTE, num_containers)

    wrap_modes = [rng.random() < 0.5 for _ in range(num_containers)]
    if num_containers >= 2 and all(wrap_modes):
        wrap_modes[rng.randrange(num_containers)] = False
    if num_containers >= 2 and not any(wrap_modes):
        wrap_modes[rng.randrange(num_containers)] = True

    specs: list[dict[str, Any]] = []
    for color, wrap in zip(colors, wrap_modes):
        inner_h = rng.randint(4, 8)
        inner_w = rng.randint(5, 10)
        plan = _build_fragment_plan(rng, inner_h, inner_w, wrap)
        if plan is None:
            return None
        pieces, anchors, filled = plan
        marker = _choose_marker(rng, filled)
        specs.append(
            {
                "color": color,
                "wrap": wrap,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "pieces": pieces,
                "anchors": anchors,
                "filled": filled,
                "marker": marker,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    for spec in sorted(specs, key=lambda s: int(s["outer_h"]) * int(s["outer_w"]), reverse=True):
        spot = _find_free_spot(
            rng,
            occupied,
            int(spec["outer_h"]),
            int(spec["outer_w"]),
            pad=1,
            tries=500,
        )
        if spot is None:
            return None
        top, left = spot
        spec["top"] = top
        spec["left"] = left
        occupied[top : top + int(spec["outer_h"]), left : left + int(spec["outer_w"])] = True

    fragment_jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for spec in specs:
        for piece in spec["pieces"]:
            fragment_jobs.append((spec, piece))
    rng.shuffle(fragment_jobs)
    fragment_jobs.sort(key=lambda job: -int(job[1]["area"]))

    for spec, piece in fragment_jobs:
        spot = _find_free_spot(
            rng,
            occupied,
            int(piece["h"]),
            int(piece["w"]),
            pad=1,
            tries=500,
        )
        if spot is None:
            return None
        top, left = spot
        piece["scatter_top"] = top
        piece["scatter_left"] = left
        occupied[top : top + int(piece["h"]), left : left + int(piece["w"])] = True

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)

    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        inner_h = int(spec["inner_h"])
        inner_w = int(spec["inner_w"])
        color = int(spec["color"])
        _paint_frame(inp, top, left, inner_h, inner_w, color)
        _paint_frame(expected, top, left, inner_h, inner_w, color)

        mr, mc = spec["marker"]
        inp[top + 1 + int(mr), left + 1 + int(mc)] = MARKER

        for piece, (pr, pc) in zip(spec["pieces"], spec["anchors"]):
            _paint_piece(expected, top + 1 + int(pr), left + 1 + int(pc), color, piece)

    for spec, piece in fragment_jobs:
        _paint_piece(
            inp,
            int(piece["scatter_top"]),
            int(piece["scatter_left"]),
            int(spec["color"]),
            piece,
        )

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
