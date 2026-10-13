from __future__ import annotations

from collections import Counter
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    clip: bool = False,
) -> None:
    h, w = patch.shape
    hh, ww = dst.shape
    rr0 = int(r0)
    cc0 = int(c0)

    if clip:
        dr0 = max(0, rr0)
        dc0 = max(0, cc0)
        dr1 = min(hh, rr0 + h)
        dc1 = min(ww, cc0 + w)
        if dr0 >= dr1 or dc0 >= dc1:
            return
        pr0 = dr0 - rr0
        pc0 = dc0 - cc0
        view = patch[pr0 : pr0 + (dr1 - dr0), pc0 : pc0 + (dc1 - dc0)]
        dest = dst[dr0:dr1, dc0:dc1]
    else:
        if rr0 < 0 or cc0 < 0 or rr0 + h > hh or cc0 + w > ww:
            raise ValueError("paste out of bounds")
        view = patch
        dest = dst[rr0 : rr0 + h, cc0 : cc0 + w]

    if transparent is None:
        dest[:, :] = view
    else:
        mask = view != int(transparent)
        dest[mask] = view[mask]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return out


def _find_foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append({"coords": np.array(coords, dtype=int), "bbox": _bbox_from_coords(coords), "area": len(coords)})
    return out


def _extract_bars(grid: np.ndarray, bg: int) -> list[dict]:
    components = _find_monochrome_components(grid, bg)
    horiz: dict[tuple[int, int], list[dict]] = {}
    vert: dict[tuple[int, int], list[dict]] = {}

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height == 1 and comp["area"] == width:
            horiz.setdefault((comp["color"], r0), []).append(comp)
        if width == 1 and comp["area"] == height:
            vert.setdefault((comp["color"], c0), []).append(comp)

    bars = []
    for (color, row), pieces in horiz.items():
        if len(pieces) < 2:
            continue
        pieces = sorted(pieces, key=lambda comp: comp["bbox"][1])
        lengths = [comp["bbox"][3] - comp["bbox"][1] + 1 for comp in pieces]
        gaps = [pieces[i + 1]["bbox"][1] - pieces[i]["bbox"][3] - 1 for i in range(len(pieces) - 1)]
        if len(set(lengths)) != 1 or len(set(gaps)) != 1 or gaps[0] <= 0:
            continue
        bars.append(
            {
                "orientation": "h",
                "color": int(color),
                "line": int(row),
                "span": (pieces[0]["bbox"][1], pieces[-1]["bbox"][3]),
                "period": int(lengths[0] + gaps[0]),
                "coords": np.concatenate([piece["coords"] for piece in pieces], axis=0),
            }
        )

    for (color, col), pieces in vert.items():
        if len(pieces) < 2:
            continue
        pieces = sorted(pieces, key=lambda comp: comp["bbox"][0])
        lengths = [comp["bbox"][2] - comp["bbox"][0] + 1 for comp in pieces]
        gaps = [pieces[i + 1]["bbox"][0] - pieces[i]["bbox"][2] - 1 for i in range(len(pieces) - 1)]
        if len(set(lengths)) != 1 or len(set(gaps)) != 1 or gaps[0] <= 0:
            continue
        bars.append(
            {
                "orientation": "v",
                "color": int(color),
                "line": int(col),
                "span": (pieces[0]["bbox"][0], pieces[-1]["bbox"][2]),
                "period": int(lengths[0] + gaps[0]),
                "coords": np.concatenate([piece["coords"] for piece in pieces], axis=0),
            }
        )

    assert bars, "expected at least one dashed control bar"
    return bars


def _nearest_unique(items: list[dict], distance_fn) -> dict:
    distances = [distance_fn(item) for item in items]
    best = min(distances)
    winners = [item for item, dist in zip(items, distances) if dist == best]
    assert len(winners) == 1, "expected a unique nearest aligned sprite"
    return winners[0]


def _matched_sprite(bar: dict, sprites: list[dict]) -> dict:
    line = bar["line"]
    span0, span1 = bar["span"]
    if bar["orientation"] == "h":
        aligned = [sprite for sprite in sprites if sprite["bbox"][1] == span0 and sprite["bbox"][3] == span1]
        candidates = [sprite for sprite in aligned if sprite["bbox"][2] < line or sprite["bbox"][0] > line]
        assert candidates, "expected a sprite aligned to the horizontal bar span"
        return _nearest_unique(
            candidates,
            lambda sprite: line - sprite["bbox"][2] if sprite["bbox"][2] < line else sprite["bbox"][0] - line,
        )

    aligned = [sprite for sprite in sprites if sprite["bbox"][0] == span0 and sprite["bbox"][2] == span1]
    candidates = [sprite for sprite in aligned if sprite["bbox"][3] < line or sprite["bbox"][1] > line]
    assert candidates, "expected a sprite aligned to the vertical bar span"
    return _nearest_unique(
        candidates,
        lambda sprite: line - sprite["bbox"][3] if sprite["bbox"][3] < line else sprite["bbox"][1] - line,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Erase dashed control bars, then stamp each aligned sprite toward its bar at the bar's own rhythm.

    Input structure:
    - The grid has a dominant background, several colored sprites, and one-cell-thick dashed bars.
    - Each dashed bar is made of equal-length painted runs separated by equal-length gaps.
    - A horizontal bar shares exactly one sprite's column span, and a vertical bar shares exactly one sprite's row span.

    Rule:
    - Read each bar's period as painted-run length plus gap length, then erase all bar cells.
    - Match every bar to the uniquely aligned sprite on one side of it.
    - Starting from the sprite's original position, stamp transparent copies every period toward the bar until a copy reaches the bar's row or column.

    Key insight:
    - The dashed bar is both the destination guide and the spacing instruction, so the copy stride comes from the bar's dash cadence rather than from the sprite size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)

    bars = _extract_bars(g, bg)
    out = g.copy()
    for bar in bars:
        coords = bar["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    sprites = _find_foreground_components(out, bg)
    assert sprites, "expected sprites after removing bars"

    for bar in bars:
        sprite = _matched_sprite(bar, sprites)
        sr0, sc0, sr1, sc1 = sprite["bbox"]
        motif = crop_bbox(out, sprite["bbox"])
        mh, mw = motif.shape
        period = int(bar["period"])
        line = int(bar["line"])

        if bar["orientation"] == "h":
            assert (sc0, sc1) == tuple(bar["span"])
            if line < sr0:
                r = sr0
                while r + mh - 1 >= line:
                    paste(out, motif, r, sc0, transparent=bg, clip=True)
                    r -= period
            else:
                r = sr0
                while r <= line:
                    paste(out, motif, r, sc0, transparent=bg, clip=True)
                    r += period
        else:
            assert (sr0, sr1) == tuple(bar["span"])
            if line < sc0:
                c = sc0
                while c + mw - 1 >= line:
                    paste(out, motif, sr0, c, transparent=bg, clip=True)
                    c -= period
            else:
                c = sc0
                while c <= line:
                    paste(out, motif, sr0, c, transparent=bg, clip=True)
                    c += period

    return out


_RNG = random.Random()
_BG = 0
_PALETTE = list(range(1, 10))
_GRID_H_RANGE = (18, 30)
_GRID_W_RANGE = (18, 30)
_PAIR_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
_MAX_ATTEMPTS = 400


def _mask_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _make_sprite_patch(height: int, width: int, color: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if height >= 3 and width >= 3:
        interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
        _RNG.shuffle(interior)
        max_holes = min(len(interior), max(1, (height * width) // 8))
        holes = _RNG.randint(0, max_holes)
        carved = 0
        for r, c in interior:
            if carved >= holes:
                break
            mask[r, c] = False
            if (
                not _mask_connected(mask)
                or not mask[0].any()
                or not mask[-1].any()
                or not mask[:, 0].any()
                or not mask[:, -1].any()
            ):
                mask[r, c] = True
            else:
                carved += 1

    patch = np.full((height, width), _BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _make_bar_patch(span: int, dash_len: int, gap_len: int, color: int, orientation: str) -> np.ndarray:
    if orientation == "h":
        patch = np.full((1, span), _BG, dtype=int)
        axis = 1
    else:
        patch = np.full((span, 1), _BG, dtype=int)
        axis = 0

    pos = 0
    while pos < span:
        end = min(span, pos + dash_len)
        if axis == 1:
            patch[0, pos:end] = color
        else:
            patch[pos:end, 0] = color
        pos += dash_len + gap_len
    return patch


def _bbox_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    if a[2] < b[0]:
        dr = b[0] - a[2] - 1
    elif b[2] < a[0]:
        dr = a[0] - b[2] - 1
    else:
        dr = 0

    if a[3] < b[1]:
        dc = b[1] - a[3] - 1
    elif b[3] < a[1]:
        dc = a[1] - b[3] - 1
    else:
        dc = 0
    return max(dr, dc)


def _gap_ok(candidate: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    return all(_bbox_gap(candidate, other) >= gap for other in others)


def _render_output_coords(spec: dict, sr0: int, sc0: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    patch = spec["sprite_patch"]
    h, w = patch.shape
    period = spec["period"]
    steps = spec["steps"]
    side = spec["side"]
    mask = patch != _BG
    hh, ww = shape
    coords: set[tuple[int, int]] = set()
    for k in range(steps + 1):
        if side == "up":
            r0 = sr0 - k * period
            c0 = sc0
        elif side == "down":
            r0 = sr0 + k * period
            c0 = sc0
        elif side == "left":
            r0 = sr0
            c0 = sc0 - k * period
        else:
            r0 = sr0
            c0 = sc0 + k * period

        for rr in range(h):
            for cc in range(w):
                if not mask[rr, cc]:
                    continue
                gr = r0 + rr
                gc = c0 + cc
                if 0 <= gr < hh and 0 <= gc < ww:
                    coords.add((gr, gc))
    return coords


def _sample_spec(sprite_color: int, bar_color: int) -> dict:
    orientation = _RNG.choice(("h", "v"))
    side = _RNG.choice(("up", "down")) if orientation == "h" else _RNG.choice(("left", "right"))
    dash_len = _RNG.randint(1, 3)
    gap_len = _RNG.randint(1, 3)
    segments = _RNG.randint(2, 4)
    span = segments * dash_len + (segments - 1) * gap_len
    period = dash_len + gap_len
    steps = _RNG.randint(1, 4)

    if orientation == "h":
        sprite_h_choices = [value for value in range(2, 6) if value != period]
        sprite_h = _RNG.choice(sprite_h_choices)
        sprite_patch = _make_sprite_patch(sprite_h, span, sprite_color)
        max_offset = min(sprite_h - 1, steps * period - 1)
        if side == "up":
            bar_offset = _RNG.randint(0, max_offset)
        else:
            min_offset = max(0, sprite_h - steps * period)
            bar_offset = _RNG.randint(min_offset, sprite_h - 1)
    else:
        sprite_w_choices = [value for value in range(2, 6) if value != period]
        sprite_w = _RNG.choice(sprite_w_choices)
        sprite_patch = _make_sprite_patch(span, sprite_w, sprite_color)
        max_offset = min(sprite_w - 1, steps * period - 1)
        if side == "left":
            bar_offset = _RNG.randint(0, max_offset)
        else:
            min_offset = max(0, sprite_w - steps * period)
            bar_offset = _RNG.randint(min_offset, sprite_w - 1)

    return {
        "orientation": orientation,
        "side": side,
        "dash_len": dash_len,
        "gap_len": gap_len,
        "segments": segments,
        "span": span,
        "period": period,
        "steps": steps,
        "bar_color": int(bar_color),
        "sprite_patch": sprite_patch,
        "bar_offset": bar_offset,
    }


def _candidate_placements(
    spec: dict,
    grid_shape: tuple[int, int],
    input_bboxes: list[tuple[int, int, int, int]],
    output_bboxes: list[tuple[int, int, int, int]],
    used_row_spans: set[tuple[int, int]],
    used_col_spans: set[tuple[int, int]],
) -> list[dict]:
    hh, ww = grid_shape
    patch = spec["sprite_patch"]
    ph, pw = patch.shape
    period = spec["period"]
    steps = spec["steps"]
    side = spec["side"]
    out = []

    if side == "up":
        row_range = range(steps * period, hh - ph + 1)
        col_range = range(0, ww - pw + 1)
    elif side == "down":
        row_range = range(0, hh - ph - steps * period + 1)
        col_range = range(0, ww - pw + 1)
    elif side == "left":
        row_range = range(0, hh - ph + 1)
        col_range = range(steps * period, ww - pw + 1)
    else:
        row_range = range(0, hh - ph + 1)
        col_range = range(0, ww - pw - steps * period + 1)

    for sr0 in row_range:
        for sc0 in col_range:
            sr1 = sr0 + ph - 1
            sc1 = sc0 + pw - 1
            row_span = (sr0, sr1)
            col_span = (sc0, sc1)
            if row_span in used_row_spans or col_span in used_col_spans:
                continue

            if side == "up":
                bar_r = sr0 - steps * period + spec["bar_offset"]
                bar_c = sc0
                input_bbox = (bar_r, sc0, sr1, sc1)
                output_bbox = (sr0 - steps * period, sc0, sr1, sc1)
            elif side == "down":
                bar_r = sr0 + steps * period + spec["bar_offset"]
                bar_c = sc0
                input_bbox = (sr0, sc0, bar_r, sc1)
                output_bbox = (sr0, sc0, sr1 + steps * period, sc1)
            elif side == "left":
                bar_r = sr0
                bar_c = sc0 - steps * period + spec["bar_offset"]
                input_bbox = (sr0, bar_c, sr1, sc1)
                output_bbox = (sr0, sc0 - steps * period, sr1, sc1)
            else:
                bar_r = sr0
                bar_c = sc0 + steps * period + spec["bar_offset"]
                input_bbox = (sr0, sc0, sr1, bar_c)
                output_bbox = (sr0, sc0, sr1, sc1 + steps * period)

            if not _gap_ok(input_bbox, input_bboxes, gap=2):
                continue
            if not _gap_ok(output_bbox, output_bboxes, gap=1):
                continue

            output_coords = _render_output_coords(spec, sr0, sc0, grid_shape)
            out.append(
                {
                    "sprite_rc": (sr0, sc0),
                    "bar_rc": (bar_r, bar_c),
                    "input_bbox": input_bbox,
                    "output_bbox": output_bbox,
                    "row_span": row_span,
                    "col_span": col_span,
                    "output_coords": output_coords,
                }
            )
    _RNG.shuffle(out)
    return out


def _draw_spec(grid: np.ndarray, spec: dict, placement: dict) -> None:
    sr0, sc0 = placement["sprite_rc"]
    paste(grid, spec["sprite_patch"], sr0, sc0, transparent=_BG)
    bar_patch = _make_bar_patch(spec["span"], spec["dash_len"], spec["gap_len"], spec["bar_color"], spec["orientation"])
    br0, bc0 = placement["bar_rc"]
    paste(grid, bar_patch, br0, bc0, transparent=_BG)


def _generate_attempt() -> np.ndarray | None:
    height = _RNG.randint(*_GRID_H_RANGE)
    width = _RNG.randint(*_GRID_W_RANGE)
    pair_count = _RNG.choice(_PAIR_COUNT_CHOICES)
    if pair_count * 2 > len(_PALETTE):
        return None

    colors = _RNG.sample(_PALETTE, pair_count * 2)
    specs = []
    for idx in range(pair_count):
        sprite_color = colors[2 * idx]
        bar_color = colors[2 * idx + 1]
        spec = _sample_spec(sprite_color, bar_color)
        if spec["sprite_patch"].shape[0] > height or spec["sprite_patch"].shape[1] > width:
            return None
        specs.append(spec)

    grid = np.full((height, width), _BG, dtype=int)
    input_bboxes: list[tuple[int, int, int, int]] = []
    output_bboxes: list[tuple[int, int, int, int]] = []
    used_row_spans: set[tuple[int, int]] = set()
    used_col_spans: set[tuple[int, int]] = set()
    placements = []

    for spec in sorted(specs, key=lambda item: max(item["sprite_patch"].shape) + item["steps"] * item["period"], reverse=True):
        candidates = _candidate_placements(spec, grid.shape, input_bboxes, output_bboxes, used_row_spans, used_col_spans)
        if not candidates:
            return None
        placement = candidates[0]
        placements.append((spec, placement))
        input_bboxes.append(placement["input_bbox"])
        output_bboxes.append(placement["output_bbox"])
        used_row_spans.add(placement["row_span"])
        used_col_spans.add(placement["col_span"])

    for spec, placement in placements:
        _draw_spec(grid, spec, placement)

    try:
        solved = main(grid)
    except Exception:
        return None

    if solved.shape != grid.shape or np.array_equal(solved, grid):
        return None

    if len(_extract_bars(grid, _BG)) != pair_count:
        return None

    if Counter(map(int, grid.ravel()))[_BG] <= Counter(map(int, grid.ravel())).most_common(2)[1][1]:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dashed-bar stamping puzzle")
