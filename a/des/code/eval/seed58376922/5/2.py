from __future__ import annotations

from collections import Counter, deque

import numpy as np

YELLOW = 4
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        neighbors = DIR4
    else:
        neighbors = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in neighbors:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _rotate_point(dr: int, dc: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return dr, dc
    if k == 1:
        return dc, -dr
    if k == 2:
        return -dr, -dc
    return -dc, dr


def _rotate_rel_cells(rel_cells: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    return [_rotate_point(dr, dc, k) for dr, dc in rel_cells]


def _find_frames(grid: np.ndarray, bg: int) -> tuple[dict[int, dict[str, object]], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = {int(v) for v in np.unique(g)} - {int(bg), YELLOW}
    frames: dict[int, dict[str, object]] = {}
    border_mask = np.zeros((h, w), dtype=bool)
    anchor_mask = np.zeros((h, w), dtype=bool)

    for color in colors:
        for coords in _components(g == color, connectivity=4):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
                continue
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            bh, bw = patch.shape
            border = np.zeros((bh, bw), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(patch == color, border):
                continue
            inner = patch[1:-1, 1:-1]
            yellow_coords = np.argwhere(inner == YELLOW)
            if yellow_coords.shape[0] != 1:
                continue
            if np.any((inner != bg) & (inner != YELLOW)):
                continue
            ar, ac = map(int, yellow_coords[0])
            anchor = (int(r0 + 1 + ar), int(c0 + 1 + ac))
            if color in frames:
                raise AssertionError(f"duplicate frame color {color}")
            frames[color] = {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "anchor": anchor,
                "inner_shape": (int(bh - 2), int(bw - 2)),
            }
            border_mask[r0 : r1 + 1, c0 : c1 + 1] |= border
            anchor_mask[anchor] = True
    return frames, border_mask, anchor_mask


def _find_fragments(
    grid: np.ndarray,
    bg: int,
    frame_border_mask: np.ndarray,
    frame_anchor_mask: np.ndarray,
) -> dict[int, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    mask = g != bg
    mask[frame_border_mask] = False
    mask[frame_anchor_mask] = False

    fragments: dict[int, dict[str, object]] = {}
    for coords in _components(mask, connectivity=4):
        values = [int(g[r, c]) for r, c in coords]
        yellow_cells = [tuple(map(int, p)) for p in coords if int(g[p[0], p[1]]) == YELLOW]
        colors = {v for v in values if v != YELLOW}
        if len(yellow_cells) != 1 or len(colors) != 1:
            continue
        color = next(iter(colors))
        anchor_r, anchor_c = yellow_cells[0]
        rel_cells = [
            (int(r - anchor_r), int(c - anchor_c))
            for r, c in coords
        ]
        if color in fragments:
            raise AssertionError(f"duplicate fragment color {color}")
        fragments[color] = {
            "coords": coords.copy(),
            "rel_cells": rel_cells,
        }
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames, frame_border_mask, frame_anchor_mask = _find_frames(g, bg)
    fragments = _find_fragments(g, bg, frame_border_mask, frame_anchor_mask)

    out = g.copy()
    for fragment in fragments.values():
        coords = fragment["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for color, frame in frames.items():
        if color not in fragments:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        anchor_r, anchor_c = frame["anchor"]
        rel_cells = fragments[color]["rel_cells"]

        chosen: list[tuple[int, int]] | None = None
        for k in range(4):
            rotated = _rotate_rel_cells(rel_cells, k)
            ok = True
            for dr, dc in rotated:
                rr = anchor_r + dr
                cc = anchor_c + dc
                if not (r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1):
                    ok = False
                    break
                if out[rr, cc] not in (bg, YELLOW):
                    ok = False
                    break
            if ok:
                chosen = rotated
                break

        if chosen is None:
            continue

        for dr, dc in chosen:
            out[anchor_r + dr, anchor_c + dc] = int(color)

    return out


COLOR_POOL = [0, 1, 2, 3, 5, 6, 7, 8, 9]
PAIR_COUNT_RANGE = (2, 5)
INNER_SIZE_RANGE = (4, 8)
SHAPE_CELL_RANGE = (3, 7)
MAX_OBJECT_ATTEMPTS = 200
MAX_LAYOUT_ATTEMPTS = 300
_RNG = np.random.default_rng()


def _sample_shape(rng: np.random.Generator) -> list[tuple[int, int]] | None:
    total = int(rng.integers(*SHAPE_CELL_RANGE))
    cells = {(0, 0)}
    while len(cells) < total:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if (nr, nc) in cells or abs(nr) > 3 or abs(nc) > 3:
                    continue
                frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(frontier[int(rng.integers(len(frontier)))])

    rotated_forms = {
        frozenset(_rotate_point(r, c, k) for r, c in cells)
        for k in range(4)
    }
    if len(rotated_forms) < 2:
        return None
    return sorted(cells)


def _candidate_anchors(
    rel_cells: list[tuple[int, int]],
    inner_h: int,
    inner_w: int,
    target_rot: int,
) -> list[tuple[int, int]]:
    rotated = [_rotate_rel_cells(rel_cells, k) for k in range(4)]
    candidates: list[tuple[int, int, int]] = []
    for ar in range(inner_h):
        for ac in range(inner_w):
            valids = []
            for k, cells in enumerate(rotated):
                if all(0 <= ar + dr < inner_h and 0 <= ac + dc < inner_w for dr, dc in cells):
                    valids.append(k)
            if target_rot in valids and min(valids) == target_rot:
                candidates.append((ar, ac, len(valids)))
    unique = [(r, c) for r, c, count in candidates if count == 1]
    if unique:
        return unique
    return [(r, c) for r, c, _count in candidates]


def _sample_pair_spec(rng: np.random.Generator, color: int, target_rot: int) -> dict[str, object] | None:
    for _ in range(MAX_OBJECT_ATTEMPTS):
        inner_h = int(rng.integers(*INNER_SIZE_RANGE))
        inner_w = int(rng.integers(*INNER_SIZE_RANGE))
        rel_cells = _sample_shape(rng)
        if rel_cells is None:
            continue
        candidates = _candidate_anchors(rel_cells, inner_h, inner_w, target_rot)
        if not candidates:
            continue
        anchor_local = candidates[int(rng.integers(len(candidates)))]
        rows = [r for r, _ in rel_cells]
        cols = [c for _, c in rel_cells]
        min_r = min(rows)
        max_r = max(rows)
        min_c = min(cols)
        max_c = max(cols)
        return {
            "color": int(color),
            "target_rot": int(target_rot),
            "rel_cells": rel_cells,
            "frame_outer_h": int(inner_h + 2),
            "frame_outer_w": int(inner_w + 2),
            "frame_anchor_local": (int(anchor_local[0]), int(anchor_local[1])),
            "frag_bbox_h": int(max_r - min_r + 1),
            "frag_bbox_w": int(max_c - min_c + 1),
            "frag_min_r": int(min_r),
            "frag_min_c": int(min_c),
        }
    return None


def _sample_rect_location(
    reserved: np.ndarray,
    box_h: int,
    box_w: int,
    pad: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    h, w = reserved.shape
    if box_h + 2 > h or box_w + 2 > w:
        return None
    max_r = h - box_h - 1
    max_c = w - box_w - 1
    if max_r < 1 or max_c < 1:
        return None
    for _ in range(500):
        r0 = int(rng.integers(1, max_r + 1))
        c0 = int(rng.integers(1, max_c + 1))
        if reserved[r0 : r0 + box_h, c0 : c0 + box_w].any():
            continue
        rr0 = max(0, r0 - pad)
        rr1 = min(h, r0 + box_h + pad)
        cc0 = max(0, c0 - pad)
        cc1 = min(w, c0 + box_w + pad)
        reserved[rr0:rr1, cc0:cc1] = True
        return r0, c0
    return None


def _paint_frame(grid: np.ndarray, spec: dict[str, object], bg: int) -> None:
    color = int(spec["color"])
    r0, c0 = spec["frame_pos"]
    fh = int(spec["frame_outer_h"])
    fw = int(spec["frame_outer_w"])
    grid[r0, c0 : c0 + fw] = color
    grid[r0 + fh - 1, c0 : c0 + fw] = color
    grid[r0 : r0 + fh, c0] = color
    grid[r0 : r0 + fh, c0 + fw - 1] = color
    ar, ac = spec["frame_anchor_local"]
    grid[r0 + 1 + ar, c0 + 1 + ac] = YELLOW


def _paint_fragment_input(grid: np.ndarray, spec: dict[str, object]) -> None:
    color = int(spec["color"])
    anchor_r, anchor_c = spec["frag_anchor_abs"]
    for dr, dc in spec["rel_cells"]:
        rr = anchor_r + dr
        cc = anchor_c + dc
        grid[rr, cc] = YELLOW if (dr, dc) == (0, 0) else color


def _paint_fragment_output(grid: np.ndarray, spec: dict[str, object]) -> None:
    color = int(spec["color"])
    frame_r0, frame_c0 = spec["frame_pos"]
    ar, ac = spec["frame_anchor_local"]
    anchor_r = frame_r0 + 1 + ar
    anchor_c = frame_c0 + 1 + ac
    for dr, dc in _rotate_rel_cells(spec["rel_cells"], int(spec["target_rot"])):
        grid[anchor_r + dr, anchor_c + dc] = color


def _build_expected(specs: list[dict[str, object]], shape: tuple[int, int], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for spec in specs:
        _paint_frame(out, spec, bg)
    for spec in specs:
        _paint_fragment_output(out, spec)
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(*PAIR_COUNT_RANGE))
    palette = rng.choice(np.asarray(COLOR_POOL, dtype=int), size=pair_count + 1, replace=False)
    bg = int(palette[0])
    colors = [int(v) for v in palette[1:]]
    target_rots = [int(v) for v in rng.permutation(4)[:pair_count]]

    specs: list[dict[str, object]] = []
    for color, target_rot in zip(colors, target_rots):
        spec = _sample_pair_spec(rng, color, target_rot)
        if spec is None:
            return None
        specs.append(spec)

    base = 18 + 3 * pair_count
    grid_h = int(rng.integers(base, base + 9))
    grid_w = int(rng.integers(base, base + 11))
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in sorted(specs, key=lambda item: int(item["frame_outer_h"]) * int(item["frame_outer_w"]), reverse=True):
        pos = _sample_rect_location(
            reserved,
            int(spec["frame_outer_h"]),
            int(spec["frame_outer_w"]),
            pad=2,
            rng=rng,
        )
        if pos is None:
            return None
        spec["frame_pos"] = pos

    for spec in sorted(specs, key=lambda item: int(item["frag_bbox_h"]) * int(item["frag_bbox_w"]), reverse=True):
        pos = _sample_rect_location(
            reserved,
            int(spec["frag_bbox_h"]),
            int(spec["frag_bbox_w"]),
            pad=1,
            rng=rng,
        )
        if pos is None:
            return None
        spec["frag_bbox_pos"] = pos
        spec["frag_anchor_abs"] = (
            int(pos[0] - int(spec["frag_min_r"])),
            int(pos[1] - int(spec["frag_min_c"])),
        )

    inp = np.full((grid_h, grid_w), int(bg), dtype=int)
    for spec in specs:
        _paint_frame(inp, spec, bg)
    for spec in specs:
        _paint_fragment_input(inp, spec)

    expected = _build_expected(specs, inp.shape, bg)

    color_counter = Counter(int(v) for v in inp.ravel())
    if color_counter[bg] <= max(color_counter[c] for c in color_counter if c != bg):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_LAYOUT_ATTEMPTS):
        grid = _generate_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
