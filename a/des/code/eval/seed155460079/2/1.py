from __future__ import annotations

from collections import deque
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_from(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = start
    assert 0 <= sr < h and 0 <= sc < w and bool(mask[sr, sc])
    q = deque([start])
    seen = {start}
    pts: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        pts.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return pts


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _parse_legend(grid: np.ndarray, bg: int) -> dict:
    h, w = grid.shape
    fg = grid != bg
    candidates: list[dict] = []
    for start in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if not bool(fg[start]):
            continue
        pts = _component_from(fg, start)
        r0, c0, r1, c1 = _bbox(pts)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        for tile_size in (3, 4):
            if sub.shape[1] != 2 * tile_size or sub.shape[0] % tile_size != 0:
                continue
            row_count = sub.shape[0] // tile_size
            if row_count < 2:
                continue
            entries = []
            ok = True
            for i in range(row_count):
                row = sub[i * tile_size : (i + 1) * tile_size, :]
                swatch = row[:, :tile_size]
                template_block = row[:, tile_size:]

                swatch_colors = [int(v) for v in np.unique(swatch) if int(v) != bg]
                if len(swatch_colors) != 1 or not np.all(swatch == swatch_colors[0]):
                    ok = False
                    break
                border_color = swatch_colors[0]

                fill_colors = [int(v) for v in np.unique(template_block) if int(v) != bg]
                if len(fill_colors) != 1:
                    ok = False
                    break
                fill_color = fill_colors[0]
                template_mask = template_block == fill_color
                if not np.any(template_mask):
                    ok = False
                    break
                entries.append(
                    {
                        "border_color": border_color,
                        "fill_color": fill_color,
                        "template": template_mask,
                    }
                )

            border_colors = [entry["border_color"] for entry in entries]
            if ok and len(set(border_colors)) == len(border_colors):
                mask = np.zeros_like(grid, dtype=bool)
                mask[r0 : r1 + 1, c0 : c1 + 1] = True
                candidates.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "tile_size": tile_size,
                        "entries": entries,
                        "mask": mask,
                    }
                )

    assert candidates, "legend not found"
    candidates.sort(key=lambda item: (len(item["entries"]), item["bbox"][2] - item["bbox"][0]), reverse=True)
    return candidates[0]


def _find_frames(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        mask = (grid == color) & (~legend_mask)
        for pts in _components(mask):
            r0, c0, r1, c1 = _bbox(pts)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 3 or ww < 3:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((hh, ww), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            color_mask = sub == color
            if int(color_mask.sum()) != int(border.sum()) - 1:
                continue
            if np.any(color_mask & (~border)):
                continue
            missing = border & (~color_mask)
            if int(missing.sum()) != 1 or not np.all(sub[missing] == bg):
                continue
            if not np.all(sub[1:-1, 1:-1] == bg):
                continue
            mr, mc = map(int, np.argwhere(missing)[0])
            if mr == 0:
                side = "top"
                inward = (1, 0)
                outer = (r0 - 1, c0 + mc)
            elif mr == hh - 1:
                side = "bottom"
                inward = (-1, 0)
                outer = (r0 + mr + 1, c0 + mc)
            elif mc == 0:
                side = "left"
                inward = (0, 1)
                outer = (r0 + mr, c0 - 1)
            else:
                side = "right"
                inward = (0, -1)
                outer = (r0 + mr, c0 + mc + 1)
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "gate": (r0 + mr, c0 + mc),
                    "outer": outer,
                    "inward": inward,
                    "side": side,
                }
            )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return frames


def _move_sprite_and_fill(
    out: np.ndarray,
    grid: np.ndarray,
    bg: int,
    frame: dict,
    template: np.ndarray,
    fill_color: int,
    legend_mask: np.ndarray,
) -> None:
    h, w = grid.shape
    outer = frame["outer"]
    assert 0 <= outer[0] < h and 0 <= outer[1] < w
    assert not bool(legend_mask[outer]) and int(grid[outer]) != bg

    sprite_color = int(grid[outer])
    sprite_mask = (grid == sprite_color) & (~legend_mask)
    sprite_cells = _component_from(sprite_mask, outer)
    dr, dc = frame["inward"]

    for r, c in sprite_cells:
        out[r, c] = bg

    moved_cells: list[tuple[int, int]] = []
    for r, c in sprite_cells:
        nr = r + 2 * dr
        nc = c + 2 * dc
        moved_cells.append((nr, nc))

    r0, c0, r1, c1 = frame["bbox"]
    for nr, nc in moved_cells:
        assert r0 < nr < r1 and c0 < nc < c1
        out[nr, nc] = sprite_color

    anchor_r = outer[0] + 2 * dr
    anchor_c = outer[1] + 2 * dc
    tile_h, tile_w = template.shape
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(out[r, c]) != bg:
                continue
            if bool(template[(r - anchor_r) % tile_h, (c - anchor_c) % tile_w]):
                out[r, c] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each seed sprite through its frame gate, then tile the frame interior
    using the legend template keyed by the frame border color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend = _parse_legend(g, bg)
    legend_mask = legend["mask"]
    entry_map = {entry["border_color"]: entry for entry in legend["entries"]}

    frames = _find_frames(g, bg, legend_mask)
    assert frames, "no frames found"
    assert set(entry_map) == {frame["color"] for frame in frames}

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend["bbox"]
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg

    for frame in frames:
        entry = entry_map[frame["color"]]
        _move_sprite_and_fill(out, g, bg, frame, entry["template"], entry["fill_color"], legend_mask)

    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_GRID_H = (22, 31)
GEN_GRID_W = (24, 33)
GEN_FRAME_COUNT = (2, 3)
GEN_TILE_SIZES = (3, 4)
GEN_INNER_H = (5, 10)
GEN_INNER_W = (5, 11)
GEN_SPRITE_LEN = (2, 4)
GEN_ATTEMPTS = 400


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = _component_from(mask, tuple(map(int, pts[0])))
    return len(seen) == int(mask.sum())


def _random_template(tile_size: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        mask = np.zeros((tile_size, tile_size), dtype=bool)
        corners = [(0, 0), (0, tile_size - 1), (tile_size - 1, 0), (tile_size - 1, tile_size - 1)]
        for r, c in corners:
            mask[r, c] = True
        for r in range(tile_size):
            for c in range(tile_size):
                if mask[r, c]:
                    continue
                if rng.random() < 0.45:
                    mask[r, c] = True
        count = int(mask.sum())
        if count < tile_size + 1 or count >= tile_size * tile_size:
            continue
        if _mask_connected(mask):
            return mask
    raise RuntimeError("failed to sample template")


def _legend_patch(entries: list[dict], tile_size: int, bg: int) -> np.ndarray:
    patch = np.full((len(entries) * tile_size, 2 * tile_size), bg, dtype=int)
    for idx, entry in enumerate(entries):
        rs = idx * tile_size
        patch[rs : rs + tile_size, :tile_size] = int(entry["border_color"])
        block = patch[rs : rs + tile_size, tile_size:]
        block[entry["template"]] = int(entry["fill_color"])
    return patch


def _place_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    entries: list[dict],
    tile_size: int,
    corner: str,
    bg: int,
) -> tuple[int, int, int, int]:
    patch = _legend_patch(entries, tile_size, bg)
    hh, ww = patch.shape
    h, w = grid.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - ww
    elif corner == "bl":
        r0, c0 = h - hh, 0
    else:
        r0, c0 = h - hh, w - ww
    grid[r0 : r0 + hh, c0 : c0 + ww] = patch
    rr0 = max(0, r0 - 2)
    cc0 = max(0, c0 - 2)
    rr1 = min(h - 1, r0 + hh + 1)
    cc1 = min(w - 1, c0 + ww + 1)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return r0, c0, r0 + hh - 1, c0 + ww - 1


def _sprite_offsets(length: int, rng: random.Random) -> list[int]:
    anchor_idx = rng.randrange(length)
    return [i - anchor_idx for i in range(length)]


def _frame_and_sprite_cells(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    side: str,
    gate_pos: int,
    offsets: list[int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], tuple[int, int]]:
    fh = inner_h + 2
    fw = inner_w + 2
    frame_cells: set[tuple[int, int]] = set()
    for cc in range(c0, c0 + fw):
        frame_cells.add((r0, cc))
        frame_cells.add((r0 + fh - 1, cc))
    for rr in range(r0, r0 + fh):
        frame_cells.add((rr, c0))
        frame_cells.add((rr, c0 + fw - 1))

    if side == "top":
        gate = (r0, c0 + gate_pos)
        sprite_cells = {(r0 - 1, c0 + gate_pos + off) for off in offsets}
    elif side == "bottom":
        gate = (r0 + fh - 1, c0 + gate_pos)
        sprite_cells = {(r0 + fh, c0 + gate_pos + off) for off in offsets}
    elif side == "left":
        gate = (r0 + gate_pos, c0)
        sprite_cells = {(r0 + gate_pos + off, c0 - 1) for off in offsets}
    else:
        gate = (r0 + gate_pos, c0 + fw - 1)
        sprite_cells = {(r0 + gate_pos + off, c0 + fw) for off in offsets}

    frame_cells.remove(gate)
    return frame_cells, sprite_cells, gate


def _gate_bounds(inner_span: int, offsets: list[int]) -> tuple[int, int]:
    lo = 1 - min(offsets)
    hi = inner_span - max(offsets)
    return lo, hi


def _oracle_output(grid: np.ndarray, bg: int, legend_bbox: tuple[int, int, int, int], specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg
    for spec in specs:
        for r, c in spec["sprite_cells"]:
            out[r, c] = bg
        for r, c in spec["moved_sprite_cells"]:
            out[r, c] = spec["sprite_color"]
        r0, c0, r1, c1 = spec["frame_bbox"]
        anchor_r, anchor_c = spec["anchor"]
        template = spec["template"]
        fill_color = spec["fill_color"]
        th, tw = template.shape
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(out[r, c]) != bg:
                    continue
                if bool(template[(r - anchor_r) % th, (c - anchor_c) % tw]):
                    out[r, c] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GEN_GRID_H)
    w = rng.randint(*GEN_GRID_W)
    bg = GEN_BG
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    frame_count = rng.randint(*GEN_FRAME_COUNT)
    tile_size = rng.choice(GEN_TILE_SIZES)

    color_pool = list(range(1, 10))
    rng.shuffle(color_pool)
    border_colors = color_pool[:frame_count]
    fill_colors = color_pool[frame_count : 2 * frame_count]
    sprite_colors = color_pool[2 * frame_count : 3 * frame_count]

    templates: list[np.ndarray] = []
    seen_templates: set[bytes] = set()
    while len(templates) < frame_count:
        mask = _random_template(tile_size, rng)
        key = mask.tobytes()
        if key in seen_templates:
            continue
        seen_templates.add(key)
        templates.append(mask)

    entries = [
        {
            "border_color": border_colors[i],
            "fill_color": fill_colors[i],
            "template": templates[i],
        }
        for i in range(frame_count)
    ]

    corner = rng.choice(["tl", "tr", "bl", "br"])
    legend_bbox = _place_legend(grid, occupied, entries, tile_size, corner, bg)

    sides = rng.sample(["top", "bottom", "left", "right"], k=frame_count)
    specs: list[dict] = []

    for idx in range(frame_count):
        side = sides[idx]
        border_color = border_colors[idx]
        fill_color = fill_colors[idx]
        sprite_color = sprite_colors[idx]
        template = templates[idx]

        placed = False
        for _ in range(200):
            inner_h = rng.randint(*GEN_INNER_H)
            inner_w = rng.randint(*GEN_INNER_W)
            sprite_len = rng.randint(*GEN_SPRITE_LEN)
            offsets = _sprite_offsets(sprite_len, rng)

            if side in ("top", "bottom"):
                if inner_w < sprite_len:
                    continue
                gate_lo, gate_hi = _gate_bounds(inner_w, offsets)
                if gate_lo > gate_hi:
                    continue
                gate_pos = rng.randint(gate_lo, gate_hi)
            else:
                if inner_h < sprite_len:
                    continue
                gate_lo, gate_hi = _gate_bounds(inner_h, offsets)
                if gate_lo > gate_hi:
                    continue
                gate_pos = rng.randint(gate_lo, gate_hi)

            fh = inner_h + 2
            fw = inner_w + 2
            r_low = 2
            c_low = 2
            r_high = h - fh - 2
            c_high = w - fw - 2
            if side == "top":
                r_low = max(r_low, 2)
            elif side == "bottom":
                r_high = min(r_high, h - fh - 3)
            elif side == "left":
                c_low = max(c_low, 2)
            else:
                c_high = min(c_high, w - fw - 3)
            if r_low > r_high or c_low > c_high:
                continue

            r0 = rng.randint(r_low, r_high)
            c0 = rng.randint(c_low, c_high)
            frame_cells, sprite_cells, gate = _frame_and_sprite_cells(r0, c0, inner_h, inner_w, side, gate_pos, offsets)
            all_cells = frame_cells | sprite_cells | {gate}

            if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
                continue

            rows = [r for r, _ in all_cells]
            cols = [c for _, c in all_cells]
            rr0 = max(0, min(rows) - 1)
            cc0 = max(0, min(cols) - 1)
            rr1 = min(h - 1, max(rows) + 1)
            cc1 = min(w - 1, max(cols) + 1)
            if np.any(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                continue

            for r, c in frame_cells:
                grid[r, c] = border_color
            for r, c in sprite_cells:
                grid[r, c] = sprite_color

            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

            if side == "top":
                outer = (gate[0] - 1, gate[1])
                inward = (1, 0)
            elif side == "bottom":
                outer = (gate[0] + 1, gate[1])
                inward = (-1, 0)
            elif side == "left":
                outer = (gate[0], gate[1] - 1)
                inward = (0, 1)
            else:
                outer = (gate[0], gate[1] + 1)
                inward = (0, -1)

            moved_sprite_cells = {(r + 2 * inward[0], c + 2 * inward[1]) for r, c in sprite_cells}
            frame_bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            fr0, fc0, fr1, fc1 = frame_bbox
            if not all(fr0 < r < fr1 and fc0 < c < fc1 for r, c in moved_sprite_cells):
                grid[:, :] = bg
                occupied[:, :] = False
                _place_legend(grid, occupied, entries, tile_size, corner, bg)
                for prev in specs:
                    for r, c in prev["frame_cells"]:
                        grid[r, c] = prev["border_color"]
                    for r, c in prev["sprite_cells"]:
                        grid[r, c] = prev["sprite_color"]
                placed = False
                break

            specs.append(
                {
                    "border_color": border_color,
                    "fill_color": fill_color,
                    "sprite_color": sprite_color,
                    "template": template,
                    "frame_bbox": frame_bbox,
                    "gate": gate,
                    "outer": outer,
                    "anchor": (outer[0] + 2 * inward[0], outer[1] + 2 * inward[1]),
                    "frame_cells": frame_cells,
                    "sprite_cells": sprite_cells,
                    "moved_sprite_cells": moved_sprite_cells,
                }
            )
            placed = True
            break

        if not placed:
            return None

    expected = _oracle_output(grid, bg, legend_bbox, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
