from __future__ import annotations

from fractions import Fraction
from itertools import permutations
import random

import numpy as np


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]


BG = 0
LEGEND = 5
PLACEHOLDER = 7
MARKER_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 400


def neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_union(boxes: list[BBox]) -> BBox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _relative_point(pt: Coord, box: BBox) -> tuple[Fraction, Fraction]:
    r, c = pt
    r0, c0, r1, c1 = box
    assert r1 > r0 and c1 > c0
    return Fraction(r - r0, r1 - r0), Fraction(c - c0, c1 - c0)


def _relative_center(box: BBox, ref_box: BBox) -> tuple[Fraction, Fraction]:
    r0, c0, r1, c1 = box
    rr0, cc0, rr1, cc1 = ref_box
    assert rr1 > rr0 and cc1 > cc0
    return (
        Fraction(r0 + r1 - 2 * rr0, 2 * (rr1 - rr0)),
        Fraction(c0 + c1 - 2 * cc0, 2 * (cc1 - cc0)),
    )


def _dist2(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> Fraction:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _unique_assignment(
    src_pos: list[tuple[Fraction, Fraction]],
    dst_pos: list[tuple[Fraction, Fraction]],
) -> tuple[int, ...]:
    scored: list[tuple[Fraction, tuple[int, ...]]] = []
    for perm in permutations(range(len(dst_pos))):
        total = sum(_dist2(src_pos[i], dst_pos[perm[i]]) for i in range(len(src_pos)))
        scored.append((total, perm))
    best = min(score for score, _ in scored)
    winners = [perm for score, perm in scored if score == best]
    assert len(winners) == 1
    return winners[0]


def _legend_and_markers(g: np.ndarray) -> tuple[BBox, list[tuple[Coord, int]]]:
    grey_comps = components_mask(g == LEGEND, connectivity=4)
    assert grey_comps
    legend_coords = max(grey_comps, key=len)
    box = bbox_coords(legend_coords)
    r0, c0, r1, c1 = box
    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch[0, :] == LEGEND)
    assert np.all(patch[-1, :] == LEGEND)
    assert np.all(patch[:, 0] == LEGEND)
    assert np.all(patch[:, -1] == LEGEND)

    markers: list[tuple[Coord, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color != LEGEND:
                markers.append(((r, c), color))
    assert 3 <= len(markers) <= 4
    marker_colors = [color for _, color in markers]
    assert len(set(marker_colors)) == len(marker_colors)
    return box, markers


def _placeholder_blocks(g: np.ndarray) -> list[BBox]:
    comps = components_mask(g == PLACEHOLDER, connectivity=4)
    boxes = [bbox_coords(coords) for coords in comps]
    for box in boxes:
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h in (2, 3) and w in (2, 3)
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == PLACEHOLDER)
    return boxes


def _inside_box(r: int, c: int, box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _isolated_pixels(g: np.ndarray, legend_box: BBox) -> list[tuple[Coord, int]]:
    r0, c0, r1, c1 = legend_box
    out: list[tuple[Coord, int]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if _inside_box(r, c, legend_box):
                continue
            color = int(g[r, c])
            if color in MARKER_COLORS:
                assert all(
                    not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]) or g[nr, nc] == BG
                    for nr, nc in neighbors8(r, c)
                )
                out.append(((r, c), color))
    return out


def _alignment(block: BBox, pixel: Coord) -> tuple[str, int] | None:
    r, c = pixel
    r0, c0, r1, c1 = block
    options: list[tuple[str, int]] = []
    if r0 <= r <= r1 and c < c0:
        options.append(("left", r))
    if r0 <= r <= r1 and c > c1:
        options.append(("right", r))
    if c0 <= c <= c1 and r < r0:
        options.append(("up", c))
    if c0 <= c <= c1 and r > r1:
        options.append(("down", c))
    assert len(options) <= 1
    return options[0] if options else None


def _path_clear(g: np.ndarray, block: BBox, pixel: Coord) -> bool:
    info = _alignment(block, pixel)
    if info is None:
        return False
    direction, axis = info
    r, c = pixel
    r0, c0, r1, c1 = block
    if direction == "left":
        return bool(np.all(g[axis, c + 1 : c0] == BG))
    if direction == "right":
        return bool(np.all(g[axis, c1 + 1 : c] == BG))
    if direction == "up":
        return bool(np.all(g[r + 1 : r0, axis] == BG))
    return bool(np.all(g[r1 + 1 : r, axis] == BG))


def _ray_geometry(shape: tuple[int, int], block: BBox, pixel: Coord) -> tuple[list[Coord], list[Coord], list[Coord]]:
    h, w = shape
    info = _alignment(block, pixel)
    assert info is not None
    direction, axis = info
    r, c = pixel
    r0, c0, r1, c1 = block
    if direction == "left":
        start = (axis, c0)
        far = (axis, c - 1)
        line = [(axis, cc) for cc in range(c0, c - 2, -1)]
        cap0 = [(axis + dr, c0) for dr in (-1, 0, 1)]
        cap1 = [(axis + dr, c - 1) for dr in (-1, 0, 1)]
    elif direction == "right":
        start = (axis, c1)
        far = (axis, c + 1)
        line = [(axis, cc) for cc in range(c1, c + 2)]
        cap0 = [(axis + dr, c1) for dr in (-1, 0, 1)]
        cap1 = [(axis + dr, c + 1) for dr in (-1, 0, 1)]
    elif direction == "up":
        start = (r0, axis)
        far = (r - 1, axis)
        line = [(rr, axis) for rr in range(r0, r - 2, -1)]
        cap0 = [(r0, axis + dc) for dc in (-1, 0, 1)]
        cap1 = [(r - 1, axis + dc) for dc in (-1, 0, 1)]
    else:
        start = (r1, axis)
        far = (r + 1, axis)
        line = [(rr, axis) for rr in range(r1, r + 2)]
        cap0 = [(r1, axis + dc) for dc in (-1, 0, 1)]
        cap1 = [(r + 1, axis + dc) for dc in (-1, 0, 1)]

    def clip(cells: list[Coord]) -> list[Coord]:
        return [(rr, cc) for rr, cc in cells if 0 <= rr < h and 0 <= cc < w]

    assert start and far
    return clip(line), clip(cap0), clip(cap1)


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor placeholder rectangles from a legend and extend same-color rays through matched isolated pixels.

    Input structure:
    - The background is black, with one grey legend rectangle containing 3 to 4 distinct one-pixel colored markers away from its border.
    - Outside the legend are the same number of solid orange placeholder rectangles plus isolated colored pixels in the marker palette.
    - Each isolated colored pixel is horizontally or vertically aligned with exactly one placeholder rectangle and the straight segment between them is otherwise background.

    Rule:
    - Match placeholder rectangles to legend markers by unique minimum total Euclidean distance between their relative positions: rectangle bbox centers are normalized inside the union bbox of all placeholders, marker cells inside the legend bbox.
    - Recolor each orange rectangle to its matched marker color.
    - For every isolated pixel whose color matches the recolored rectangle it aligns with, draw a 1-cell-thick ray from the nearest rectangle edge through that pixel and one cell beyond, add a 3-cell perpendicular cap at both ray ends, then erase the original isolated-pixel cell back to black.
    - Leave the grey legend body, its internal markers, and every non-matching isolated pixel unchanged.

    Key insight:
    - The visible isolated-pixel colors are insufficient by themselves because alignment alone is ambiguous; the legend first assigns each rectangle a color, and only that color ownership tells which aligned pixels trigger ray extension.
    """
    g = np.asarray(grid, dtype=int)
    legend_box, markers = _legend_and_markers(g)
    blocks = _placeholder_blocks(g)
    assert len(blocks) == len(markers)

    union_box = bbox_union(blocks)
    block_pos = [_relative_center(box, union_box) for box in blocks]
    marker_pos = [_relative_point(pos, legend_box) for pos, _ in markers]
    assignment = _unique_assignment(block_pos, marker_pos)
    block_colors = [markers[j][1] for j in assignment]

    isolated = _isolated_pixels(g, legend_box)
    pixel_block: dict[Coord, int] = {}
    for pixel, _color in isolated:
        matches = [
            i
            for i, box in enumerate(blocks)
            if _alignment(box, pixel) is not None and _path_clear(g, box, pixel)
        ]
        assert len(matches) == 1
        pixel_block[pixel] = matches[0]

    out = g.copy()
    for box, color in zip(blocks, block_colors):
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = color

    for pixel, color in isolated:
        idx = pixel_block[pixel]
        if color != block_colors[idx]:
            continue
        line, cap0, cap1 = _ray_geometry(g.shape, blocks[idx], pixel)
        for rr, cc in line + cap0 + cap1:
            out[rr, cc] = color
        out[pixel] = BG

    return out


def _boxes_separated(a: BBox, b: BBox, gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _fill_box(g: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    g[r0 : r1 + 1, c0 : c1 + 1] = color


def _rows_and_cols_unique(boxes: list[BBox]) -> tuple[list[list[int]], list[list[int]], list[list[tuple[str, int]]]]:
    unique_rows: list[list[int]] = []
    unique_cols: list[list[int]] = []
    line_opts: list[list[tuple[str, int]]] = []
    for i, box in enumerate(boxes):
        r0, c0, r1, c1 = box
        rows = [r for r in range(r0, r1 + 1) if sum(ob[0] <= r <= ob[2] for ob in boxes) == 1]
        cols = [c for c in range(c0, c1 + 1) if sum(ob[1] <= c <= ob[3] for ob in boxes) == 1]
        unique_rows.append(rows)
        unique_cols.append(cols)
        opts = [("h", r) for r in rows] + [("v", c) for c in cols]
        line_opts.append(opts)
    return unique_rows, unique_cols, line_opts


def _plan_targets(line_opts: list[list[tuple[str, int]]]) -> tuple[list[dict], dict[int, int]] | None:
    n = len(line_opts)
    line_counts = [len(opts) for opts in line_opts]
    if any(count == 0 for count in line_counts):
        return None

    true_counts = [1] * n
    rich = [i for i, count in enumerate(line_counts) if count >= 2]
    double_block = None
    if rich and random.random() < 0.45:
        double_block = random.choice(rich)
        true_counts[double_block] = 2

    free_slots = {i: line_counts[i] - true_counts[i] for i in range(n)}
    recipient_candidates = [i for i, cap in free_slots.items() if cap >= 1]
    color_candidates = [i for i in range(n) if true_counts[i] == 1]
    if not recipient_candidates or not color_candidates:
        return None

    max_decoys = min(len(recipient_candidates), len(color_candidates))
    want = max(1, min(max_decoys, random.randint(1, max_decoys)))
    if max_decoys >= n - 1 and random.random() < 0.5:
        want = min(max_decoys, n - 1)

    decoy_map: dict[int, int] = {}
    tries = 0
    while tries < 120:
        tries += 1
        chosen_colors = random.sample(color_candidates, want)
        chosen_recipients = random.sample(recipient_candidates, want)
        ok_perms = [
            perm
            for perm in permutations(chosen_recipients)
            if all(color != perm[k] for k, color in enumerate(chosen_colors))
        ]
        if not ok_perms:
            continue
        perm = random.choice(ok_perms)
        decoy_map = {color: recipient for color, recipient in zip(chosen_colors, perm)}
        break
    if not decoy_map:
        return None

    tasks: list[dict] = []
    for i in range(n):
        for _ in range(true_counts[i]):
            tasks.append({"block": i, "source": i, "kind": "true"})
    for src, dst in decoy_map.items():
        tasks.append({"block": dst, "source": src, "kind": "decoy"})

    need_per_block = [0] * n
    for task in tasks:
        need_per_block[task["block"]] += 1
    if any(need > line_counts[i] for i, need in enumerate(need_per_block)):
        return None
    return tasks, true_counts


def _marker_cells(legend_box: BBox, n: int) -> list[Coord] | None:
    r0, c0, r1, c1 = legend_box
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    for _ in range(120):
        random.shuffle(interior)
        chosen: list[Coord] = []
        for cell in interior:
            if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) >= 2 for rr, cc in chosen):
                chosen.append(cell)
                if len(chosen) == n:
                    rows = {r for r, _ in chosen}
                    cols = {c for _, c in chosen}
                    if len(rows) >= 2 and len(cols) >= 2:
                        return chosen
                    break
    return None


def _candidate_pixels(shape: tuple[int, int], box: BBox, line: tuple[str, int]) -> list[Coord]:
    h, w = shape
    r0, c0, r1, c1 = box
    kind, axis = line
    out: list[Coord] = []
    if kind == "h":
        max_left = c0
        for dist in range(2, min(7, max_left + 1)):
            out.append((axis, c0 - dist))
        if max_left >= 1:
            out.append((axis, 0))
        max_right = w - 1 - c1
        for dist in range(2, min(7, max_right + 1)):
            out.append((axis, c1 + dist))
        if max_right >= 1:
            out.append((axis, w - 1))
    else:
        max_up = r0
        for dist in range(2, min(7, max_up + 1)):
            out.append((r0 - dist, axis))
        if max_up >= 1:
            out.append((0, axis))
        max_down = h - 1 - r1
        for dist in range(2, min(7, max_down + 1)):
            out.append((r1 + dist, axis))
        if max_down >= 1:
            out.append((h - 1, axis))
    random.shuffle(out)
    return out


def _static_masks(shape: tuple[int, int], legend_box: BBox, blocks: list[BBox]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    legend_mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = legend_box
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    block_mask = np.zeros((h, w), dtype=bool)
    for box in blocks:
        rr0, cc0, rr1, cc1 = box
        block_mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return legend_mask, block_mask


def _can_place_pixel(
    g: np.ndarray,
    legend_mask: np.ndarray,
    block_mask: np.ndarray,
    blocks: list[BBox],
    block_idx: int,
    pixel: Coord,
    used_output: set[Coord],
    kind: str,
) -> bool:
    r, c = pixel
    h, w = g.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if g[r, c] != BG:
        return False
    if pixel in used_output:
        return False
    for nr, nc in neighbors8(r, c):
        if 0 <= nr < h and 0 <= nc < w and g[nr, nc] != BG:
            return False
    box = blocks[block_idx]
    if _alignment(box, pixel) is None or not _path_clear(g, box, pixel):
        return False
    if kind == "decoy":
        return True

    line, cap0, cap1 = _ray_geometry(g.shape, box, pixel)
    allowed_box = box
    for rr, cc in set(line + cap0 + cap1):
        if (rr, cc) == pixel:
            continue
        if (rr, cc) in used_output:
            return False
        if legend_mask[rr, cc]:
            return False
        if block_mask[rr, cc] and not _inside_box(rr, cc, allowed_box):
            return False
        if g[rr, cc] != BG and not _inside_box(rr, cc, allowed_box):
            return False
    return True


def _backtrack_pixels(
    g: np.ndarray,
    legend_mask: np.ndarray,
    block_mask: np.ndarray,
    blocks: list[BBox],
    tasks: list[dict],
    line_opts: list[list[tuple[str, int]]],
    colors: list[int],
    idx: int,
    used_lines: list[set[tuple[str, int]]],
    used_output: set[Coord],
    placements: list[dict],
) -> bool:
    if idx == len(tasks):
        return True

    task = tasks[idx]
    block_idx = task["block"]
    color = colors[task["source"]]
    candidate_lines = [line for line in line_opts[block_idx] if line not in used_lines[block_idx]]
    random.shuffle(candidate_lines)
    for line in candidate_lines:
        for pixel in _candidate_pixels(g.shape, blocks[block_idx], line):
            if not _can_place_pixel(g, legend_mask, block_mask, blocks, block_idx, pixel, used_output, task["kind"]):
                continue

            g[pixel] = color
            used_lines[block_idx].add(line)
            added_output: set[Coord] = set()
            if task["kind"] == "true":
                line_cells, cap0, cap1 = _ray_geometry(g.shape, blocks[block_idx], pixel)
                added_output = {cell for cell in line_cells + cap0 + cap1 if cell != pixel}
                used_output |= added_output
            placements.append(
                {
                    "block": block_idx,
                    "pixel": pixel,
                    "color": color,
                    "kind": task["kind"],
                    "source": task["source"],
                    "line": line,
                }
            )
            if _backtrack_pixels(
                g,
                legend_mask,
                block_mask,
                blocks,
                tasks,
                line_opts,
                colors,
                idx + 1,
                used_lines,
                used_output,
                placements,
            ):
                return True
            placements.pop()
            if task["kind"] == "true":
                used_output -= added_output
            used_lines[block_idx].remove(line)
            g[pixel] = BG
    return False


def _place_tasks_greedy(
    g: np.ndarray,
    legend_mask: np.ndarray,
    block_mask: np.ndarray,
    blocks: list[BBox],
    tasks: list[dict],
    line_opts: list[list[tuple[str, int]]],
    colors: list[int],
) -> list[dict] | None:
    ordered = sorted(tasks, key=lambda t: (t["kind"] != "true", len(line_opts[t["block"]])))
    for _ in range(80):
        temp = g.copy()
        placements: list[dict] = []
        used_lines = [set() for _ in blocks]
        used_output: set[Coord] = set()
        ok = True
        for task in ordered:
            block_idx = task["block"]
            color = colors[task["source"]]
            candidates: list[tuple[tuple[str, int], Coord]] = []
            lines = [line for line in line_opts[block_idx] if line not in used_lines[block_idx]]
            random.shuffle(lines)
            for line in lines:
                for pixel in _candidate_pixels(temp.shape, blocks[block_idx], line):
                    if _can_place_pixel(temp, legend_mask, block_mask, blocks, block_idx, pixel, used_output, task["kind"]):
                        candidates.append((line, pixel))
            if not candidates:
                ok = False
                break
            line, pixel = random.choice(candidates)
            temp[pixel] = color
            used_lines[block_idx].add(line)
            if task["kind"] == "true":
                ray, cap0, cap1 = _ray_geometry(temp.shape, blocks[block_idx], pixel)
                used_output |= {cell for cell in ray + cap0 + cap1 if cell != pixel}
            placements.append(
                {
                    "block": block_idx,
                    "pixel": pixel,
                    "color": color,
                    "kind": task["kind"],
                    "source": task["source"],
                    "line": line,
                }
            )
        if ok:
            g[:, :] = temp
            return placements
    return None


def _render_from_plan(inp: np.ndarray, blocks: list[BBox], block_colors: list[int], placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    for box, color in zip(blocks, block_colors):
        r0, c0, r1, c1 = box
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    for item in placements:
        if item["kind"] != "true":
            continue
        color = block_colors[item["block"]]
        line, cap0, cap1 = _ray_geometry(inp.shape, blocks[item["block"]], item["pixel"])
        for rr, cc in line + cap0 + cap1:
            out[rr, cc] = color
        out[item["pixel"]] = BG
    return out


def _validate_semantics(inp: np.ndarray, plan: dict) -> bool:
    legend_box, markers = _legend_and_markers(inp)
    blocks = _placeholder_blocks(inp)
    if len(blocks) != len(markers):
        return False

    isolated = _isolated_pixels(inp, legend_box)
    if len(isolated) != len(plan["placements"]):
        return False

    counts: dict[int, int] = {}
    true_by_block = [0] * len(blocks)
    wrong_aligned = 0
    for pixel, color in isolated:
        counts[color] = counts.get(color, 0) + 1
        matches = [i for i, box in enumerate(blocks) if _alignment(box, pixel) is not None and _path_clear(inp, box, pixel)]
        if len(matches) != 1:
            return False
        block_idx = matches[0]
        if color == plan["block_colors"][block_idx]:
            true_by_block[block_idx] += 1
        else:
            wrong_aligned += 1
    if any(v not in (1, 2) for v in counts.values()):
        return False
    if wrong_aligned < 1:
        return False
    if not all(v >= 1 for v in true_by_block):
        return False

    union_box = bbox_union(blocks)
    block_pos = [_relative_center(box, union_box) for box in blocks]
    marker_pos = [_relative_point(pos, legend_box) for pos, _ in markers]
    assignment = _unique_assignment(block_pos, marker_pos)
    colors = [markers[j][1] for j in assignment]
    if colors != plan["block_colors"]:
        return False

    expected = _render_from_plan(inp, blocks, colors, plan["placements"])
    actual = main(inp)
    return np.array_equal(expected, actual)


def _sample_legend_box(h: int, w: int) -> BBox:
    lh = random.randint(6, 8)
    lw = random.randint(6, 8)
    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        r0 = random.randint(1, 3)
        c0 = random.randint(1, 3)
    elif corner == "tr":
        r0 = random.randint(1, 3)
        c0 = w - lw - random.randint(1, 3)
    elif corner == "bl":
        r0 = h - lh - random.randint(1, 3)
        c0 = random.randint(1, 3)
    else:
        r0 = h - lh - random.randint(1, 3)
        c0 = w - lw - random.randint(1, 3)
    return r0, c0, r0 + lh - 1, c0 + lw - 1


def _sample_blocks(shape: tuple[int, int], legend_box: BBox, n: int) -> list[BBox] | None:
    h, w = shape
    row_slots = list(range(1, h - 3, 4))
    col_slots = list(range(1, w - 3, 4))
    if len(row_slots) < n or len(col_slots) < n:
        return None

    for _ in range(120):
        rows = random.sample(row_slots, n)
        cols = random.sample(col_slots, n)
        perm = random.sample(range(n), n)
        boxes: list[BBox] = []
        for i in range(n):
            bh = random.randint(2, 3)
            bw = random.randint(2, 3)
            r0 = rows[i]
            c0 = cols[perm[i]]
            box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
            if box[2] >= h - 1 or box[3] >= w - 1:
                break
            if not _boxes_separated(box, legend_box, gap=1):
                break
            if any(not _boxes_separated(box, prev, gap=1) for prev in boxes):
                break
            boxes.append(box)
        if len(boxes) != n:
            continue
        if max(b[0] for b in boxes) - min(b[0] for b in boxes) < 4:
            continue
        if max(b[1] for b in boxes) - min(b[1] for b in boxes) < 4:
            continue
        return boxes
    return None


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    n = random.randint(3, 4)

    g = np.zeros((h, w), dtype=int)
    legend_box = _sample_legend_box(h, w)
    _fill_box(g, legend_box, LEGEND)

    blocks = _sample_blocks((h, w), legend_box, n)
    if blocks is None:
        return None
    for box in blocks:
        _fill_box(g, box, PLACEHOLDER)

    marker_cells = _marker_cells(legend_box, n)
    if marker_cells is None:
        return None
    marker_colors = random.sample(list(MARKER_COLORS), n)
    for cell, color in zip(marker_cells, marker_colors):
        g[cell] = color

    union_box = bbox_union(blocks)
    block_pos = [_relative_center(box, union_box) for box in blocks]
    marker_pos = [_relative_point(cell, legend_box) for cell in marker_cells]
    try:
        assignment = _unique_assignment(block_pos, marker_pos)
    except AssertionError:
        return None
    block_colors = [marker_colors[j] for j in assignment]

    _, _, line_opts = _rows_and_cols_unique(blocks)
    planned = _plan_targets(line_opts)
    if planned is None:
        return None
    tasks, true_counts = planned

    legend_mask, block_mask = _static_masks(g.shape, legend_box, blocks)
    placements = _place_tasks_greedy(g, legend_mask, block_mask, blocks, tasks, line_opts, block_colors)
    if placements is None:
        return None

    plan = {
        "blocks": blocks,
        "block_colors": block_colors,
        "placements": placements,
        "true_counts": true_counts,
    }
    if not _validate_semantics(g, plan):
        return None
    return g, plan


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            grid, _plan = sample
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
