from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
SEP_COLOR = 5
FRAME_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


TEMPLATE_MASKS_3 = [
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
]

TEMPLATE_MASKS_4 = [
    np.array([[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 0, 0], [0, 1, 1, 1], [0, 1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 0, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool),
]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*points)
    return min(rows), min(cols), max(rows), max(cols)


def _find_separator(grid: np.ndarray) -> int:
    candidates = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if row[0] != BG and np.all(row == row[0]):
            if r > 0 and r < grid.shape[0] - 1:
                candidates.append(r)
    assert len(candidates) == 1, f"expected one separator row, found {candidates}"
    return candidates[0]


def _extract_legends(lower: np.ndarray) -> dict[int, np.ndarray]:
    assert lower.shape[0] >= 4, "legend zone must contain key row and templates"
    key_row = lower[0]
    keys = [(c, int(key_row[c])) for c in range(lower.shape[1]) if int(key_row[c]) != BG]
    assert keys, "legend key row must contain at least one color key"

    template_parts = []
    for pts in _components(lower[1:] != BG):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h in (3, 4) and w in (3, 4), f"unexpected legend template size {(h, w)}"
        template_parts.append(
            {
                "bbox": (r0, c0, r1, c1),
                "template": lower[1 + r0 : 1 + r1 + 1, c0 : c1 + 1].copy(),
            }
        )

    legends: dict[int, np.ndarray] = {}
    for col, key_color in keys:
        matches = [part for part in template_parts if part["bbox"][1] <= col <= part["bbox"][3]]
        assert len(matches) == 1, f"key at column {col} should match exactly one legend template"
        legends[key_color] = matches[0]["template"]
    return legends


def _frame_components(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in sorted(set(map(int, np.unique(grid))) - {BG}):
        mask = grid == color
        for pts in _components(mask):
            r0, c0, r1, c1 = _bbox(pts)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            sub = mask[r0 : r1 + 1, c0 : c1 + 1]
            if (
                np.all(sub[0, :])
                and np.all(sub[-1, :])
                and np.all(sub[:, 0])
                and np.all(sub[:, -1])
                and np.any(~sub[1:-1, 1:-1])
            ):
                frames.append((color, (r0, c0, r1, c1)))
    assert frames, "expected at least one hollow rectangular frame"
    return frames


def _anchor_offsets(template: np.ndarray) -> tuple[int, int]:
    return template.shape[0] // 2, template.shape[1] // 2


def _stamp_inside(
    out: np.ndarray,
    template: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    box: tuple[int, int, int, int],
) -> None:
    r0, c0, r1, c1 = box
    ar, ac = _anchor_offsets(template)
    for tr in range(template.shape[0]):
        for tc in range(template.shape[1]):
            value = int(template[tr, tc])
            if value == BG:
                continue
            rr = anchor_r + tr - ar
            cc = anchor_c + tc - ac
            if r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1:
                out[rr, cc] = value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each frame's same-color seed dots with the legend template keyed by that frame border color.

    Input structure:
    - The background is black.
    - The upper part contains several hollow rectangular frames, each drawn in one color and containing one or more isolated seed dots of that same color.
    - A full gray horizontal separator splits the main area from a lower legend area.
    - In the legend area, each template has a single color-key pixel touching the separator side, and the key color identifies which frame color should use that template.

    Rule:
    - Find the separator and discard everything below it in the output.
    - Build a mapping from each key color to its nearby 3x3 or 4x4 legend template.
    - For every frame, clear its interior seed dots, then for each seed dot paste the matching template with the seed acting as the template anchor; only non-black template pixels are painted and any part crossing the frame border is clipped away.
    - Keep the original frame borders unchanged.

    Key insight:
    - The border color is a lookup key into the legend, while the interior dots are placement anchors for the retrieved template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator(g)
    upper = g[:sep].copy()
    legends = _extract_legends(g[sep + 1 :])

    for color, box in _frame_components(upper):
        assert color in legends, f"missing legend template for frame color {color}"
        r0, c0, r1, c1 = box
        anchors = list(map(tuple, np.argwhere(upper[r0 + 1 : r1, c0 + 1 : c1] == color)))
        anchors = [(r + r0 + 1, c + c0 + 1) for r, c in anchors]

        interior = upper[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == color] = BG
        for ar, ac in anchors:
            _stamp_inside(upper, legends[color], ar, ac, box)
    return upper


def _transform_template(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    out = arr.copy()
    if rng.random() < 0.5:
        out = np.fliplr(out)
    k = rng.randrange(4)
    if k:
        out = np.rot90(out, k)
    return out


def _make_template(key_color: int, frame_colors: set[int], rng: random.Random) -> np.ndarray:
    size = rng.choice([3, 4])
    base = rng.choice(TEMPLATE_MASKS_3 if size == 3 else TEMPLATE_MASKS_4)
    mask = _transform_template(base, rng)

    available = [c for c in FRAME_COLOR_POOL if c != key_color and c not in frame_colors]
    if len(available) < 3:
        available = [c for c in FRAME_COLOR_POOL if c != key_color]
    color_count = rng.randint(1, min(3, len(available)))
    palette = rng.sample(available, color_count)

    out = np.zeros(mask.shape, dtype=int)
    cells = list(map(tuple, np.argwhere(mask)))
    rng.shuffle(cells)
    for idx, (r, c) in enumerate(cells):
        if idx < len(palette):
            out[r, c] = palette[idx]
        else:
            out[r, c] = rng.choice(palette)
    return out


def _footprint(template: np.ndarray, anchor: tuple[int, int], inner_h: int, inner_w: int) -> set[tuple[int, int]]:
    ar, ac = _anchor_offsets(template)
    rr, cc = anchor
    cells: set[tuple[int, int]] = set()
    for tr in range(template.shape[0]):
        for tc in range(template.shape[1]):
            if int(template[tr, tc]) == BG:
                continue
            ir = rr + tr - ar
            ic = cc + tc - ac
            if 0 <= ir < inner_h and 0 <= ic < inner_w:
                cells.add((ir, ic))
    return cells


def _bbox_with_margin(points: set[tuple[int, int]], margin: int) -> tuple[int, int, int, int]:
    rows, cols = zip(*points)
    return min(rows) - margin, min(cols) - margin, max(rows) + margin, max(cols) + margin


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _pick_anchors(template: np.ndarray, inner_h: int, inner_w: int, rng: random.Random) -> list[tuple[int, int]]:
    total_nonzero = int(np.count_nonzero(template))
    edge_candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
    interior_candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []

    for r in range(inner_h):
        for c in range(inner_w):
            stamp = _footprint(template, (r, c), inner_h, inner_w)
            if not stamp:
                continue
            if len(stamp) == total_nonzero:
                interior_candidates.append(((r, c), stamp))
            else:
                edge_candidates.append(((r, c), stamp))

    assert interior_candidates or edge_candidates
    want = rng.randint(1, 3)
    anchors: list[tuple[int, int]] = []
    used_boxes: list[tuple[int, int, int, int]] = []
    force_edge_first = bool(edge_candidates) and rng.random() < 0.5

    for idx in range(want):
        if idx == 0 and force_edge_first:
            pools = [edge_candidates, interior_candidates]
        else:
            pools = [interior_candidates, edge_candidates] if rng.random() < 0.5 else [edge_candidates, interior_candidates]

        picked = None
        for pool in pools:
            shuffled = pool[:]
            rng.shuffle(shuffled)
            for anchor, stamp in shuffled:
                box = _bbox_with_margin(stamp, 1)
                if any(_boxes_overlap(box, old) for old in used_boxes):
                    continue
                picked = (anchor, box)
                break
            if picked is not None:
                break
        if picked is None:
            break
        anchors.append(picked[0])
        used_boxes.append(picked[1])

    assert anchors
    return anchors


def _place_frames(
    grid: np.ndarray,
    main_h: int,
    frame_specs: list[dict],
    rng: random.Random,
) -> bool:
    occupied: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(frame_specs)), key=lambda i: frame_specs[i]["h"] * frame_specs[i]["w"], reverse=True)
    for idx in order:
        spec = frame_specs[idx]
        h = spec["h"]
        w = spec["w"]
        placed = False
        for _ in range(250):
            r0 = rng.randint(0, main_h - h)
            c0 = rng.randint(0, grid.shape[1] - w)
            box = (r0 - 1, c0 - 1, r0 + h, c0 + w)
            if any(_boxes_overlap(box, old) for old in occupied):
                continue
            color = spec["color"]
            r1 = r0 + h - 1
            c1 = c0 + w - 1
            grid[r0, c0 : c1 + 1] = color
            grid[r1, c0 : c1 + 1] = color
            grid[r0 : r1 + 1, c0] = color
            grid[r0 : r1 + 1, c1] = color
            for ar, ac in spec["anchors"]:
                grid[r0 + 1 + ar, c0 + 1 + ac] = color
            occupied.append(box)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_colors = rng.randint(2, 4)
    legend_colors = rng.sample(FRAME_COLOR_POOL, num_colors)
    frame_color_set = set(legend_colors)
    templates = {color: _make_template(color, frame_color_set, rng) for color in legend_colors}

    num_frames = rng.randint(num_colors, min(4, num_colors + 1))
    frame_colors = legend_colors[:]
    while len(frame_colors) < num_frames:
        frame_colors.append(rng.choice(legend_colors))
    rng.shuffle(frame_colors)

    frame_specs = []
    for color in frame_colors:
        template = templates[color]
        inner_h = rng.randint(max(6, template.shape[0] + 2), 9)
        inner_w = rng.randint(max(6, template.shape[1] + 2), 10)
        anchors = _pick_anchors(template, inner_h, inner_w, rng)
        frame_specs.append(
            {
                "color": color,
                "anchors": anchors,
                "h": inner_h + 2,
                "w": inner_w + 2,
            }
        )

    legend_order = legend_colors[:]
    rng.shuffle(legend_order)
    legend_templates = [templates[color] for color in legend_order]
    legend_gaps = [rng.randint(1, 3) for _ in range(len(legend_order) + 1)]
    need_w = max(sum(legend_gaps) + sum(t.shape[1] for t in legend_templates), max(spec["w"] for spec in frame_specs) + 2, 18)
    if need_w > 29:
        return None

    grid_w = rng.randint(need_w, 29)
    main_h = rng.randint(max(max(spec["h"] for spec in frame_specs) + 1, 12), 19)
    max_template_h = max(t.shape[0] for t in legend_templates)
    total_h = main_h + 4 + max_template_h
    if total_h > 30:
        return None

    grid = np.zeros((total_h, grid_w), dtype=int)
    if not _place_frames(grid, main_h, frame_specs, rng):
        return None

    sep = main_h
    grid[sep, :] = SEP_COLOR
    legend_top = sep + 3
    x = 0
    for idx, color in enumerate(legend_order):
        template = templates[color]
        x += legend_gaps[idx]
        if x + template.shape[1] > grid_w - 2:
            return None
        key_col = x + _anchor_offsets(template)[1]
        grid[sep + 1, key_col] = color
        grid[legend_top : legend_top + template.shape[0], x : x + template.shape[1]] = template
        x += template.shape[1]
    if grid_w - x < legend_gaps[-1]:
        return None
    if np.any(grid[sep + 2] != BG):
        return None

    out = main(grid)
    if out.shape[0] != main_h or out.shape[1] != grid_w:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
