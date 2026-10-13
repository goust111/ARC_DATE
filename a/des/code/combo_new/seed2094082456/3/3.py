import random
from collections import Counter

import numpy as np


BG_COLOR = 0
MASK_COLOR = 5
FRAME_COLOR = 8
COLOR_POOL = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRIES = 400
OBJECT_PLACE_RETRIES = 200

BASE_SHAPES = (
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [1, 0, 1, 0], [1, 0, 1, 1], [0, 0, 1, 0]], dtype=bool),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    sub[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if np.any(sub[1:-1, 1:-1]):
        return False
    return np.all(sub[0, :]) and np.all(sub[-1, :]) and np.all(sub[:, 0]) and np.all(sub[:, -1])


def _parse_legend(grid: np.ndarray, bg: int):
    h, w = grid.shape
    clear_mask = np.zeros_like(grid, dtype=bool)
    entries = []
    for coords in _components(grid == FRAME_COLOR):
        if not _is_rect_frame(coords):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if r0 != 0 and r1 != h - 1:
            continue
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_colors = set(map(int, np.unique(inner))) - {bg}
        assert len(inner_colors) == 1, "each legend entry must contain exactly one template color"
        src_color = next(iter(inner_colors))
        template_mask = inner == src_color
        assert np.any(template_mask), "legend template cannot be empty"

        marker_candidates = []
        if c0 - 1 >= 0:
            for rr in range(r0, r1 + 1):
                val = int(grid[rr, c0 - 1])
                if val not in {bg, FRAME_COLOR}:
                    marker_candidates.append((rr, c0 - 1, val))
        if c1 + 1 < w:
            for rr in range(r0, r1 + 1):
                val = int(grid[rr, c1 + 1])
                if val not in {bg, FRAME_COLOR}:
                    marker_candidates.append((rr, c1 + 1, val))
        assert len(marker_candidates) == 1, "each legend entry must have one adjacent output-color marker"
        mr, mc, out_color = marker_candidates[0]

        clear_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        clear_mask[mr, mc] = True
        entries.append(
            {
                "src": int(src_color),
                "out": int(out_color),
                "mask": template_mask.copy(),
            }
        )
    assert entries, "legend must contain at least one entry"
    return entries, clear_mask


def main(grid: np.ndarray) -> np.ndarray:
    """Restore grey-covered shape cells by matching each damaged object to the edge dictionary and recoloring only the hidden part.

    Input structure:
    - The background is the dominant color.
    - The main area contains several small foreground shapes; some shapes have part of their cells replaced by grey mask cells.
    - Along the top or bottom edge there is a row of framed dictionary entries, each containing one full template shape and one adjacent single-pixel output-color marker.

    Rule:
    - Read every edge dictionary entry as a mapping from a source-color template shape to an output color.
    - Remove the dictionary from the working canvas, then find each connected component that contains grey mask cells.
    - Match that component's full silhouette and visible source color to the unique template with the same source color; repaint only the grey-covered cells with the template's output color, while leaving the visible source-colored cells unchanged.

    Key insight:
    - The grey cells mark missing shape pixels, but the replacement color does not come from the damaged object itself; it comes from the matching dictionary entry on the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    entries, clear_mask = _parse_legend(g, bg)

    body = g.copy()
    body[clear_mask] = bg
    out = body.copy()

    for coords in _components(body != bg):
        r0, c0, r1, c1 = _bbox(coords)
        sub = body[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(sub[sub != bg])))
        if MASK_COLOR not in colors:
            continue
        src_colors = colors - {MASK_COLOR}
        assert len(src_colors) == 1, "masked object should have one visible source color"
        src_color = next(iter(src_colors))
        occupied = sub != bg
        hidden = sub == MASK_COLOR
        matches = [
            entry
            for entry in entries
            if entry["src"] == src_color
            and entry["mask"].shape == occupied.shape
            and np.array_equal(entry["mask"], occupied)
        ]
        assert len(matches) == 1, "each masked object must match exactly one dictionary template"
        out[r0 : r1 + 1, c0 : c1 + 1][hidden] = matches[0]["out"]

    return out


def _transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return np.rot90(out, rng.randrange(4)).copy()


def _choose_templates(rng: random.Random, count: int) -> list[np.ndarray] | None:
    for _ in range(50):
        base_ids = rng.sample(range(len(BASE_SHAPES)), count)
        masks = [_transform_mask(BASE_SHAPES[idx], rng) for idx in base_ids]
        keys = {(mask.shape, mask.tobytes()) for mask in masks}
        if len(keys) == count:
            return masks
    return None


def _assign_colors(rng: random.Random, template_count: int):
    group_sizes = [2, 1] if template_count == 3 else random.choice(([2, 1, 1], [2, 2]))
    source_colors = rng.sample(COLOR_POOL, len(group_sizes))
    remaining = [c for c in COLOR_POOL if c not in source_colors]
    output_colors = rng.sample(remaining, template_count)
    src_list: list[int] = []
    for color, size in zip(source_colors, group_sizes):
        src_list.extend([color] * size)
    rng.shuffle(src_list)
    return src_list, output_colors


def _sample_hidden(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    coords = list(zip(*np.where(mask)))
    area = len(coords)
    max_hidden = max(1, min(area - 2, (2 * area) // 3))
    min_hidden = max(1, area // 4)
    hidden_count = rng.randint(min_hidden, max_hidden)
    chosen = rng.sample(coords, hidden_count)
    hidden = np.zeros_like(mask, dtype=bool)
    for r, c in chosen:
        hidden[r, c] = True
    assert 0 < hidden_count < area
    assert np.count_nonzero(mask & ~hidden) >= 2
    return hidden


def _legend_layout(entries: list[dict]) -> tuple[int, int]:
    widths = [entry["mask"].shape[1] + 3 for entry in entries]
    heights = [entry["mask"].shape[0] + 2 for entry in entries]
    total_w = sum(widths) + max(0, len(widths) - 1)
    total_h = max(heights)
    return total_h, total_w


def _place_legend(grid: np.ndarray, entries: list[dict], side: str, start_col: int):
    h, _ = grid.shape
    cursor = start_col
    for entry in entries:
        mask = entry["mask"]
        src = entry["src"]
        out = entry["out"]
        marker_side = entry["marker_side"]
        mh, mw = mask.shape
        fh, fw = mh + 2, mw + 2
        fr0 = 0 if side == "top" else h - fh
        fc0 = cursor + (1 if marker_side == "left" else 0)
        grid[fr0, fc0 : fc0 + fw] = FRAME_COLOR
        grid[fr0 + fh - 1, fc0 : fc0 + fw] = FRAME_COLOR
        grid[fr0 : fr0 + fh, fc0] = FRAME_COLOR
        grid[fr0 : fr0 + fh, fc0 + fw - 1] = FRAME_COLOR
        inner = grid[fr0 + 1 : fr0 + 1 + mh, fc0 + 1 : fc0 + 1 + mw]
        inner[mask] = src
        mr = fr0 + fh // 2
        mc = fc0 - 1 if marker_side == "left" else fc0 + fw
        grid[mr, mc] = out
        cursor += mw + 3 + 1


def _place_objects(
    grid: np.ndarray,
    objects: list[dict],
    body_r0: int,
    body_r1: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    occupancy = np.zeros((h, w), dtype=bool)
    for obj in sorted(objects, key=lambda item: item["mask"].sum(), reverse=True):
        mask = obj["mask"]
        hidden = obj["hidden"]
        color = obj["src"]
        oh, ow = mask.shape
        top_min = body_r0 + 1
        top_max = body_r1 - oh
        left_min = 1
        left_max = w - ow - 1
        if top_min > top_max or left_min > left_max:
            return False
        placed = False
        for _ in range(OBJECT_PLACE_RETRIES):
            top = rng.randint(top_min, top_max)
            left = rng.randint(left_min, left_max)
            if np.any(occupancy[top - 1 : top + oh + 1, left - 1 : left + ow + 1]):
                continue
            patch = grid[top : top + oh, left : left + ow]
            patch[mask] = color
            patch[hidden] = MASK_COLOR
            occupancy[top : top + oh, left : left + ow] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _build_objects(entries: list[dict], rng: random.Random) -> list[dict]:
    by_source: dict[int, list[int]] = {}
    for idx, entry in enumerate(entries):
        by_source.setdefault(entry["src"], []).append(idx)
    repeated_groups = [group for group in by_source.values() if len(group) > 1]
    assert repeated_groups
    masked_indices = list(rng.choice(repeated_groups))
    others = [idx for idx in range(len(entries)) if idx not in masked_indices]
    extra_masked = rng.randint(0, min(2, len(others)))
    if extra_masked:
        masked_indices.extend(rng.sample(others, extra_masked))

    objects = []
    for idx in masked_indices:
        mask = entries[idx]["mask"]
        objects.append({"mask": mask, "hidden": _sample_hidden(mask, rng), "src": entries[idx]["src"]})

    if rng.random() < 0.5:
        dup_idx = rng.choice(masked_indices)
        dup_mask = entries[dup_idx]["mask"]
        objects.append({"mask": dup_mask, "hidden": _sample_hidden(dup_mask, rng), "src": entries[dup_idx]["src"]})

    plain_count = rng.randint(0, min(2, len(entries)))
    for idx in rng.sample(range(len(entries)), plain_count):
        mask = entries[idx]["mask"]
        objects.append({"mask": mask, "hidden": np.zeros_like(mask, dtype=bool), "src": entries[idx]["src"]})

    return objects


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_count = rng.randint(3, 4)
    masks = _choose_templates(rng, template_count)
    if masks is None:
        return None

    src_colors, out_colors = _assign_colors(rng, template_count)
    entries = []
    for idx, mask in enumerate(masks):
        entries.append(
            {
                "mask": mask,
                "src": src_colors[idx],
                "out": out_colors[idx],
                "marker_side": rng.choice(("left", "right")),
            }
        )

    legend_h, legend_w = _legend_layout(entries)
    min_w = max(legend_w, 18)
    min_h = legend_h + 10
    if min_w > 30 or min_h > 30:
        return None

    w = rng.randint(min_w, 30)
    h = rng.randint(min_h, 30)
    side = rng.choice(("top", "bottom"))
    grid = np.full((h, w), BG_COLOR, dtype=int)

    start_col = rng.randint(0, w - legend_w)
    _place_legend(grid, entries, side, start_col)

    if side == "top":
        body_r0, body_r1 = legend_h + 1, h - 1
    else:
        body_r0, body_r1 = 0, h - legend_h - 2
    if body_r1 - body_r0 + 1 < 6:
        return None

    objects = _build_objects(entries, rng)
    if not _place_objects(grid, objects, body_r0, body_r1, rng):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
