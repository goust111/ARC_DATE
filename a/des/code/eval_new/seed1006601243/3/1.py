from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
CANVAS_LIMIT = 30
GEN_RETRIES = 400
OBJECT_COUNT_CHOICES = (1, 2, 3)
INTERIOR_SIZE_RANGE = (4, 7)
FRAME_COLORS = (5, 6, 7, 8)
COLOR_POOL = tuple(range(1, 10))
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = DIRS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


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
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(int(x) for x in mask.shape), mask.astype(np.uint8).tobytes()


def _frame_border_mask(h: int, w: int) -> np.ndarray:
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _find_frames(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    frame_sets: dict[int, list[dict[str, object]]] = {}
    for color in map(int, np.unique(g)):
        if color == BG:
            continue
        frames: list[dict[str, object]] = []
        for comp in _components(g == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            border = _frame_border_mask(h, w)
            if not np.array_equal(sub == color, border):
                continue
            inner = sub[1:-1, 1:-1]
            inner_colors = set(map(int, np.unique(inner))) - {BG}
            if len(inner_colors) != 1:
                continue
            inner_color = next(iter(inner_colors))
            anchor_mask = inner == inner_color
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "inner_color": int(inner_color),
                    "anchor_mask": anchor_mask,
                    "shape": anchor_mask.shape,
                }
            )
        if frames:
            frame_sets[int(color)] = frames

    assert frame_sets, "no frames found"
    frame_color, frames = max(frame_sets.items(), key=lambda item: len(item[1]))
    inner_colors = [int(frame["inner_color"]) for frame in frames]
    assert len(set(inner_colors)) == len(inner_colors), "frame contents must use unique colors"
    return int(frame_color), frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each outside template so its leaked pixels line up with the matching framed anchors, then stamp the full template into the frame.

    Input structure:
    - The background is black.
    - Several same-colored rectangular frames contain a few scattered pixels of a single non-frame color.
    - Outside the frames, each such color also appears as one standalone monochrome template on black background.

    Rule:
    - Detect every rectangular frame and read the single anchor color plus anchor positions inside its interior.
    - Match each frame to the outside template of the same color, test the four 90-degree rotations, and keep the unique rotation whose filled cells contain all anchor positions.
    - Leave the outside templates and frame borders unchanged, but replace each frame interior with the full rotated template mask.

    Key insight:
    - The colored pixels inside a frame are not the final picture; they are a rotation-specific subset of a same-colored outside template, so color finds the partner and subset containment fixes the orientation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _frame_color, frames = _find_frames(g)

    outside_mask = np.ones(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        outside_mask[r0 : r1 + 1, c0 : c1 + 1] = False

    templates: dict[int, np.ndarray] = {}
    for frame in frames:
        color = int(frame["inner_color"])
        if color in templates:
            continue
        pts = np.argwhere((g == color) & outside_mask)
        assert pts.size, f"missing template for color {color}"
        tr0, tc0, tr1, tc1 = _bbox(pts)
        templates[color] = g[tr0 : tr1 + 1, tc0 : tc1 + 1] == color

    for frame in frames:
        color = int(frame["inner_color"])
        template = templates[color]
        anchor_mask = np.asarray(frame["anchor_mask"], dtype=bool)

        matches: list[np.ndarray] = []
        seen = set()
        for k in range(4):
            rotated = np.rot90(template, k)
            if rotated.shape != anchor_mask.shape:
                continue
            if np.all(rotated[anchor_mask]):
                key = _rotation_key(rotated)
                if key in seen:
                    continue
                seen.add(key)
                matches.append(rotated)

        assert len(matches) == 1, f"rotation not uniquely determined for color {color}"
        fill_mask = matches[0]
        r0, c0, r1, c1 = frame["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[:] = BG
        inner[fill_mask] = color

    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _generate_template_mask(rng: random.Random, size: int) -> np.ndarray | None:
    target_min = max(5, size + 1)
    target_max = min(size * size - 1, size * size // 2 + 3)
    for _ in range(120):
        target = rng.randint(target_min, target_max)
        mask = np.zeros((size, size), dtype=bool)

        seeds = [
            (0, rng.randrange(size)),
            (size - 1, rng.randrange(size)),
            (rng.randrange(size), 0),
            (rng.randrange(size), size - 1),
            (rng.randrange(size), rng.randrange(size)),
        ]
        for r, c in seeds:
            mask[r, c] = True

        while int(mask.sum()) < target:
            cells = np.argwhere(mask)
            if rng.random() < 0.82:
                cr, cc = map(int, cells[rng.randrange(len(cells))])
                dr, dc = rng.choice(DIRS8)
                nr, nc = cr + dr, cc + dc
            else:
                nr, nc = rng.randrange(size), rng.randrange(size)
            if 0 <= nr < size and 0 <= nc < size:
                mask[nr, nc] = True

        if not _touches_all_sides(mask):
            continue
        if int(mask.sum()) >= size * size:
            continue
        if len({_rotation_key(np.rot90(mask, k)) for k in range(4)}) != 4:
            continue
        return mask
    return None


def _choose_anchor_subset(
    rng: random.Random, variants: list[np.ndarray], true_rot: int
) -> np.ndarray | None:
    target = variants[true_rot]
    cells = [tuple(map(int, p)) for p in np.argwhere(target)]
    remaining = {k for k in range(4) if k != true_rot and variants[k].shape == target.shape}
    chosen: list[tuple[int, int]] = []
    pool = cells[:]
    while remaining:
        best_cell = None
        best_cover: set[int] = set()
        for cell in pool:
            cover = {k for k in remaining if not bool(variants[k][cell])}
            if len(cover) > len(best_cover):
                best_cover = cover
                best_cell = cell
        if best_cell is None or not best_cover:
            return None
        chosen.append(best_cell)
        pool.remove(best_cell)
        remaining -= best_cover

    if len(chosen) >= len(cells):
        return None

    min_target = max(2, len(chosen))
    max_target = min(len(cells) - 1, max(min_target, len(cells) // 2))
    desired = rng.randint(min_target, max_target)
    extras = [cell for cell in cells if cell not in chosen]
    rng.shuffle(extras)
    for cell in extras:
        if len(chosen) >= desired:
            break
        chosen.append(cell)

    anchor = np.zeros_like(target, dtype=bool)
    for r, c in chosen:
        anchor[r, c] = True
    return anchor


def _can_place(occ: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(CANVAS_LIMIT, top + h + pad)
    c1 = min(CANVAS_LIMIT, left + w + pad)
    return not bool(occ[r0:r1, c0:c1].any())


def _place_rectangles(
    rng: random.Random, dims: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    occ = np.zeros((CANVAS_LIMIT, CANVAS_LIMIT), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(dims)
    order = sorted(range(len(dims)), key=lambda i: dims[i][0] * dims[i][1], reverse=True)
    for idx in order:
        h, w = dims[idx]
        placed = False
        for _ in range(500):
            top = rng.randint(0, CANVAS_LIMIT - h)
            left = rng.randint(0, CANVAS_LIMIT - w)
            if not _can_place(occ, top, left, h, w, pad=1):
                continue
            occ[top : top + h, left : left + w] = True
            positions[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in positions if pos is not None]


def _draw_template(canvas: np.ndarray, top: int, left: int, color: int, mask: np.ndarray) -> None:
    sub = canvas[top : top + mask.shape[0], left : left + mask.shape[1]]
    sub[mask] = color


def _draw_frame_input(
    canvas: np.ndarray,
    top: int,
    left: int,
    frame_color: int,
    inner_color: int,
    anchor_mask: np.ndarray,
) -> None:
    h, w = anchor_mask.shape
    canvas[top : top + h + 2, left : left + w + 2] = BG
    canvas[top, left : left + w + 2] = frame_color
    canvas[top + h + 1, left : left + w + 2] = frame_color
    canvas[top : top + h + 2, left] = frame_color
    canvas[top : top + h + 2, left + w + 1] = frame_color
    canvas[top + 1 : top + h + 1, left + 1 : left + w + 1][anchor_mask] = inner_color


def _draw_frame_output(
    canvas: np.ndarray,
    top: int,
    left: int,
    frame_color: int,
    inner_color: int,
    fill_mask: np.ndarray,
) -> None:
    _draw_frame_input(canvas, top, left, frame_color, inner_color, np.zeros_like(fill_mask, dtype=bool))
    canvas[top + 1 : top + fill_mask.shape[0] + 1, left + 1 : left + fill_mask.shape[1] + 1][fill_mask] = inner_color


def _crop_with_margin(
    rng: random.Random, inp: np.ndarray, out: np.ndarray, boxes: Iterable[tuple[int, int, int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    boxes = list(boxes)
    r0 = min(b[0] for b in boxes)
    c0 = min(b[1] for b in boxes)
    r1 = max(b[2] for b in boxes)
    c1 = max(b[3] for b in boxes)
    top_margin = rng.randint(0, min(3, r0))
    left_margin = rng.randint(0, min(3, c0))
    bottom_margin = rng.randint(0, min(3, CANVAS_LIMIT - 1 - r1))
    right_margin = rng.randint(0, min(3, CANVAS_LIMIT - 1 - c1))
    rr0 = r0 - top_margin
    cc0 = c0 - left_margin
    rr1 = r1 + bottom_margin
    cc1 = c1 + right_margin
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1], out[rr0 : rr1 + 1, cc0 : cc1 + 1]


def _build_instance(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_objects = rng.choice(OBJECT_COUNT_CHOICES)
    frame_color = rng.choice(FRAME_COLORS)
    template_colors = rng.sample([c for c in COLOR_POOL if c != frame_color], n_objects)

    specs: list[dict[str, object]] = []
    dims: list[tuple[int, int]] = []
    for color in template_colors:
        size = rng.randint(*INTERIOR_SIZE_RANGE)
        template = _generate_template_mask(rng, size)
        if template is None:
            return None
        variants = [np.rot90(template, k) for k in range(4)]
        true_rot = rng.randrange(4)
        anchor_mask = _choose_anchor_subset(rng, variants, true_rot)
        if anchor_mask is None:
            return None
        fill_mask = variants[true_rot]
        specs.append(
            {
                "color": int(color),
                "template": template,
                "anchor_mask": anchor_mask,
                "fill_mask": fill_mask,
                "frame_h": size + 2,
                "frame_w": size + 2,
                "template_h": size,
                "template_w": size,
            }
        )
        dims.append((size + 2, size + 2))
        dims.append((size, size))

    positions = _place_rectangles(rng, dims)
    if positions is None:
        return None

    inp = np.full((CANVAS_LIMIT, CANVAS_LIMIT), BG, dtype=int)
    out = np.full((CANVAS_LIMIT, CANVAS_LIMIT), BG, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    for idx, spec in enumerate(specs):
        frame_top, frame_left = positions[2 * idx]
        temp_top, temp_left = positions[2 * idx + 1]
        color = int(spec["color"])
        template = np.asarray(spec["template"], dtype=bool)
        anchor_mask = np.asarray(spec["anchor_mask"], dtype=bool)
        fill_mask = np.asarray(spec["fill_mask"], dtype=bool)

        _draw_frame_input(inp, frame_top, frame_left, frame_color, color, anchor_mask)
        _draw_frame_output(out, frame_top, frame_left, frame_color, color, fill_mask)
        _draw_template(inp, temp_top, temp_left, color, template)
        _draw_template(out, temp_top, temp_left, color, template)

        boxes.append((frame_top, frame_left, frame_top + template.shape[0] + 1, frame_left + template.shape[1] + 1))
        boxes.append((temp_top, temp_left, temp_top + template.shape[0] - 1, temp_left + template.shape[1] - 1))

    inp, out = _crop_with_margin(rng, inp, out, boxes)
    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > CANVAS_LIMIT or max(out.shape) > CANVAS_LIMIT:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        built = _build_instance(rng)
        if built is not None:
            inp, _ = built
            return inp
    raise RuntimeError("generate_input: exhausted attempts")
