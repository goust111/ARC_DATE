from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 5
BORDER = 0
HIGHLIGHT = 2
INPUT_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    neighbors = DIRS4 if connectivity == 4 else DIRS8
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((int(factor), int(factor)), dtype=int))


def _frame_border_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = _components_mask(grid == BORDER, connectivity=4)
    frames: list[tuple[int, int, int, int]] = []
    for coords in comps:
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        ring = set(map(tuple, coords.tolist()))
        expected = set(_frame_border_cells(box))
        if ring != expected:
            continue
        frames.append(box)
    frames.sort()
    return frames


def _find_template_sprite(grid: np.ndarray, frames: Iterable[tuple[int, int, int, int]]) -> np.ndarray:
    frame_boxes = list(frames)
    mask = ~np.isin(grid, [BG, BORDER])
    candidates = []
    for coords in _components_mask(mask, connectivity=8):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        inside_frame = any(fr0 < r0 <= r1 < fr1 and fc0 < c0 <= c1 < fc1 for fr0, fc0, fr1, fc1 in frame_boxes)
        if inside_frame:
            continue
        colors = tuple(sorted({int(grid[r, c]) for r, c in coords}))
        candidates.append((len(colors), coords.shape[0], box, colors))
    assert candidates, "expected a visible template object"
    max_colors = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == max_colors]
    assert len(best) == 1, "template color variety must be unique"
    return _crop_bbox(grid, best[0][2])


def _matching_transforms(template: np.ndarray, inner: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    observed = inner != BG
    matches: list[tuple[int, int, np.ndarray]] = []
    for scale in (2, 3):
        scaled = _scale_sprite(template, scale)
        if scaled.shape != inner.shape:
            continue
        for rot in range(4):
            transformed = np.rot90(scaled, rot)
            if np.array_equal(transformed[observed], inner[observed]):
                matches.append((scale, rot, transformed))
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each framed fragment with the matching scaled-rotated template copy, then highlight the most incomplete frame.

    Input structure:
    - The background is grey.
    - A small multicolor template sits alone in one corner.
    - Elsewhere there are 2 or 3 black 1-pixel rectangular frames whose interiors contain partial fragments of scaled 2x/3x, rotated copies of the template.

    Rule:
    - Identify the template as the non-grey object outside the frames that uses the most distinct colors.
    - For each frame interior, test every 2x/3x scaling and every 90-degree rotation of the template, and keep the unique transform whose non-grey pixels exactly agree with all visible fragment pixels.
    - Fill every missing non-grey pixel of that matched transform inside the frame and count how many pixels were added.
    - Recolor the border of the frame with the largest completion count from black to red; leave the other borders black and leave the corner template unchanged.

    Key insight:
    - The completion itself produces the signal: the solver must both reconstruct each damaged scaled template and compare the number of restored pixels to decide which border changes color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    assert 2 <= len(frames) <= 3, "expected 2 or 3 framed fragments"
    template = _find_template_sprite(g, frames)

    completions: list[tuple[int, tuple[int, int, int, int]]] = []
    for box in frames:
        r0, c0, r1, c1 = box
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        matches = _matching_transforms(template, inner)
        assert len(matches) == 1, "fragment must determine a unique scale and rotation"
        _scale, _rot, transformed = matches[0]
        fill_count = int(np.count_nonzero((inner == BG) & (transformed != BG)))
        out[r0 + 1 : r1, c0 + 1 : c1] = transformed
        completions.append((fill_count, box))

    values = [count for count, _box in completions]
    assert len(values) == len(set(values)), "completion counts must be distinct"
    highlight_box = max(completions, key=lambda item: item[0])[1]
    for r, c in _frame_border_cells(highlight_box):
        out[r, c] = HIGHLIGHT
    return out


def _random_template() -> np.ndarray:
    for _ in range(400):
        size = random.choice([3, 4])
        n_colors = random.choice([2, 3])
        palette = random.sample(INPUT_COLORS, n_colors)
        area = size * size
        min_fg = max(size + 1, n_colors + 2)
        fg_count = random.randint(min_fg, area - 1)
        sprite = np.full((size, size), BG, dtype=int)
        positions = random.sample(range(area), fg_count)
        colors = palette.copy()
        extra = fg_count - n_colors
        colors.extend(random.choices(palette, k=extra))
        random.shuffle(colors)
        for idx, color in zip(positions, colors):
            sprite[idx // size, idx % size] = color
        if len(_components_mask(sprite != BG, connectivity=8)) != 1:
            continue
        if _bbox(np.argwhere(sprite != BG)) != (0, 0, size - 1, size - 1):
            continue
        rotations = [np.rot90(sprite, k) for k in range(4)]
        if len({rot.tobytes() for rot in rotations}) != 4:
            continue
        return sprite
    raise RuntimeError("failed to build a usable template")


def _sample_partial(template: np.ndarray, scale: int, rotation: int, forbidden_counts: set[int]) -> tuple[np.ndarray, int]:
    full = np.rot90(_scale_sprite(template, scale), rotation)
    coords = np.argwhere(full != BG)
    total_fg = int(coords.shape[0])
    full_colors = sorted({int(full[r, c]) for r, c in coords})
    template_color_count = len(sorted({int(v) for v in template.flat if int(v) != BG}))

    for _ in range(500):
        used_color_count = random.randint(1, max(1, template_color_count - 1))
        chosen_colors = set(random.sample(full_colors, used_color_count))
        eligible = [tuple(map(int, rc)) for rc in coords if int(full[rc[0], rc[1]]) in chosen_colors]
        if not eligible:
            continue

        preferred_missing = []
        for ratio in (0.35, 0.5, 0.65, 0.8):
            missing = int(round(total_fg * ratio))
            if 1 <= missing < total_fg and missing not in preferred_missing:
                preferred_missing.append(missing)
        all_missing = preferred_missing + [m for m in range(1, total_fg) if m not in preferred_missing]

        for missing in all_missing:
            if missing in forbidden_counts:
                continue
            observed_count = total_fg - missing
            if observed_count < 1 or observed_count > len(eligible):
                continue
            observed = random.sample(eligible, observed_count)
            partial = np.full_like(full, BG)
            for r, c in observed:
                partial[r, c] = int(full[r, c])
            matches = _matching_transforms(template, partial)
            if len(matches) != 1:
                continue
            matched_scale, matched_rot, _ = matches[0]
            if matched_scale != scale or matched_rot != rotation:
                continue
            return partial, missing
    raise RuntimeError("failed to sample a unique partial fragment")


def _draw_frame(canvas: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    for r, c in _frame_border_cells(box):
        canvas[r, c] = int(color)


def _free_rect(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    H, W = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _mark_rect(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    H, W = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(H, top + h + pad)
    c1 = min(W, left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _place_objects(template: np.ndarray, frame_sizes: list[tuple[int, int]]) -> tuple[int, int, tuple[int, int], list[tuple[int, int]]]:
    th, tw = template.shape
    corners = ["tl", "tr", "bl", "br"]
    for _ in range(500):
        H = random.randint(max(20, max(h for h, _ in frame_sizes) + 6), 30)
        W = random.randint(max(20, max(w for _, w in frame_sizes) + 6), 30)
        occupied = np.zeros((H, W), dtype=bool)
        corner = random.choice(corners)
        if corner == "tl":
            tpos = (1, 1)
        elif corner == "tr":
            tpos = (1, W - tw - 1)
        elif corner == "bl":
            tpos = (H - th - 1, 1)
        else:
            tpos = (H - th - 1, W - tw - 1)
        _mark_rect(occupied, tpos[0], tpos[1], th, tw, pad=2)

        positions: list[tuple[int, int] | None] = [None] * len(frame_sizes)
        ok = True
        order = sorted(range(len(frame_sizes)), key=lambda idx: frame_sizes[idx], reverse=True)
        for idx in order:
            h, w = frame_sizes[idx]
            choices = [
                (r, c)
                for r in range(1, H - h)
                for c in range(1, W - w)
                if _free_rect(occupied, r, c, h, w, pad=2)
            ]
            if not choices:
                ok = False
                break
            pos = random.choice(choices)
            positions[idx] = pos
            _mark_rect(occupied, pos[0], pos[1], h, w, pad=2)
        if ok:
            return H, W, tpos, [pos for pos in positions if pos is not None]
    raise RuntimeError("failed to place template and frames")


def _reconstruct_metadata(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[tuple[int, tuple[int, int, int, int], int, int]]]:
    frames = _find_frames(grid)
    template = _find_template_sprite(grid, frames)
    details: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    for box in frames:
        r0, c0, r1, c1 = box
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        matches = _matching_transforms(template, inner)
        assert len(matches) == 1
        scale, rot, transformed = matches[0]
        fill_count = int(np.count_nonzero((inner == BG) & (transformed != BG)))
        details.append((fill_count, box, scale, rot))
    return template, frames, details


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        template = _random_template()
        frame_count = random.choice([2, 3])
        transforms = [(random.choice([2, 3]), random.randrange(4)) for _ in range(frame_count)]
        fulls = [np.rot90(_scale_sprite(template, scale), rot) for scale, rot in transforms]
        frame_sizes = [(full.shape[0] + 2, full.shape[1] + 2) for full in fulls]

        try:
            H, W, template_pos, frame_positions = _place_objects(template, frame_sizes)
        except RuntimeError:
            continue

        partials = []
        used_counts: set[int] = set()
        try:
            for (scale, rot) in transforms:
                partial, fill_count = _sample_partial(template, scale, rot, used_counts)
                partials.append((partial, fill_count))
                used_counts.add(fill_count)
        except RuntimeError:
            continue

        inp = np.full((H, W), BG, dtype=int)
        out = np.full((H, W), BG, dtype=int)

        tr, tc = template_pos
        inp[tr : tr + template.shape[0], tc : tc + template.shape[1]] = template
        out[tr : tr + template.shape[0], tc : tc + template.shape[1]] = template

        highlight_index = max(range(frame_count), key=lambda i: partials[i][1])
        for idx, ((scale, rot), full, (partial, _fill_count), (fr, fc)) in enumerate(
            zip(transforms, fulls, partials, frame_positions)
        ):
            fh, fw = full.shape
            box = (fr, fc, fr + fh + 1, fc + fw + 1)
            border_color = HIGHLIGHT if idx == highlight_index else BORDER
            _draw_frame(out, box, border_color)
            _draw_frame(inp, box, BORDER)
            out[fr + 1 : fr + 1 + fh, fc + 1 : fc + 1 + fw] = full
            inp[fr + 1 : fr + 1 + fh, fc + 1 : fc + 1 + fw] = partial

        solved = main(inp)
        if not np.array_equal(solved, out):
            continue

        recovered_template, frames, details = _reconstruct_metadata(inp)
        if not np.array_equal(recovered_template, template):
            continue
        fill_counts = [item[0] for item in details]
        if len(fill_counts) != len(set(fill_counts)):
            continue
        if max(fill_counts) != partials[highlight_index][1]:
            continue
        if len(frames) != frame_count:
            continue
        if {item[2] for item in details} - {2, 3}:
            continue
        return inp, out
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case()[0]
