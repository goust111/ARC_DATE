from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    assert counts, "expected a nonempty grid"
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty mask"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0)


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        if start >= sep:
            return []
        spans.append((int(start), int(sep) - 1))
        start = int(sep) + 1
    if start >= length:
        return []
    spans.append((int(start), int(length) - 1))
    return spans


def _parse_panel(panel: np.ndarray, panel_origin: tuple[int, int], bg: int, panel_id: int) -> dict | None:
    colors = [int(color) for color in np.unique(panel) if int(color) != int(bg)]
    if len(colors) != 1:
        return None
    color = colors[0]
    mask = panel == color
    comps = _components4(mask)
    if len(comps) != 1:
        return None
    cropped, (r0, c0) = _crop_mask(mask)
    return {
        "id": int(panel_id),
        "panel_r0": int(panel_origin[0]),
        "panel_c0": int(panel_origin[1]),
        "bbox_r0": int(r0),
        "bbox_c0": int(c0),
        "abs_r0": int(panel_origin[0]) + int(r0),
        "abs_c0": int(panel_origin[1]) + int(c0),
        "color": int(color),
        "area": int(cropped.sum()),
        "mask": cropped.astype(bool),
    }


def _parse_scene_with_separator(g: np.ndarray, bg: int, sep_color: int) -> list[dict] | None:
    h, w = g.shape
    rows = [r for r in range(h) if np.all(g[r, :] == int(sep_color))]
    cols = [c for c in range(w) if np.all(g[:, c] == int(sep_color))]
    if not rows and not cols:
        return None

    row_spans = _spans(h, rows)
    col_spans = _spans(w, cols)
    if not row_spans or not col_spans or len(row_spans) * len(col_spans) < 2:
        return None

    objects: list[dict] = []
    panel_id = 0
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panel = g[r0 : r1 + 1, c0 : c1 + 1]
            obj = _parse_panel(panel, (r0, c0), bg, panel_id)
            if obj is None:
                return None
            objects.append(obj)
            panel_id += 1
    return objects


def _extract_scene(g: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = _mode_color(g)
    candidates: list[tuple[int, list[dict]]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        parsed = _parse_scene_with_separator(g, bg, color)
        if parsed is not None:
            candidates.append((color, parsed))
    assert len(candidates) == 1, "expected a unique separator color"
    sep_color, objects = candidates[0]
    return int(bg), int(sep_color), objects


def _embedding_positions(big_mask: np.ndarray, small_mask: np.ndarray) -> list[tuple[int, int]]:
    bh, bw = map(int, big_mask.shape)
    sh, sw = map(int, small_mask.shape)
    if sh > bh or sw > bw:
        return []
    positions: list[tuple[int, int]] = []
    for dr in range(bh - sh + 1):
        for dc in range(bw - sw + 1):
            region = big_mask[dr : dr + sh, dc : dc + sw]
            if bool(np.all(region[small_mask])):
                positions.append((int(dr), int(dc)))
    return positions


def _match_small_to_big(objects: list[dict]) -> list[tuple[dict, dict, int, int]]:
    area_levels = sorted({int(obj["area"]) for obj in objects})
    valid: list[tuple[tuple[int, int], list[tuple[dict, dict, int, int]]]] = []

    for split in area_levels[:-1]:
        smalls = [obj for obj in objects if int(obj["area"]) <= int(split)]
        bigs = [obj for obj in objects if int(obj["area"]) > int(split)]
        if not smalls or not bigs:
            continue
        if max(int(obj["area"]) for obj in smalls) >= min(int(obj["area"]) for obj in bigs):
            continue

        matches: list[tuple[dict, dict, int, int]] = []
        ok = True
        for small in smalls:
            found: list[tuple[dict, int, int]] = []
            for big in bigs:
                for dr, dc in _embedding_positions(big["mask"], small["mask"]):
                    found.append((big, int(dr), int(dc)))
            if len(found) != 1:
                ok = False
                break
            big, dr, dc = found[0]
            matches.append((small, big, dr, dc))
        if ok:
            matched_big_ids = {int(big["id"]) for _small, big, _dr, _dc in matches}
            score = (len(smalls), len(matched_big_ids))
            valid.append((score, matches))

    assert valid, "could not separate panels into unique small/big groups"
    best_score = max(score for score, _matches in valid)
    best = [matches for score, matches in valid if score == best_score]
    assert len(best) == 1, "ambiguous small/big split"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the uniquely matching subshape inside each large panel using the small panel's color.

    Input structure:
    - The background is the dominant color, and one other color forms complete
      separator rows and/or columns that partition the grid into rectangular panels.
    - Every panel contains exactly one monochrome 4-connected object on the
      background, with no other foreground pixels inside the panel.
    - Some panel objects are strictly smaller by area than the others.

    Rule:
    - Parse the separator rows/columns to recover the panel layout and extract the
      single object from each panel.
    - Split the objects into a small group and a large group so that every small
      object has area strictly below every large object.
    - For each small object, find the unique large object that contains the same
      shape under translation only; recolor exactly that translated subset inside
      the large panel with the small object's color.
    - Leave all separator lines, the small panels, and the rest of each large
      object unchanged.

    Key insight:
    - The separators define the first-level structure, but the real pairing comes
      from shape containment: each small panel is a uniquely placeable fragment of
      one large panel, and its own color is transferred onto that fragment.
    """
    g = np.asarray(grid, dtype=int)
    _bg, _sep, objects = _extract_scene(g)
    matches = _match_small_to_big(objects)

    out = g.copy()
    for small, big, dr, dc in matches:
        rr, cc = np.where(small["mask"])
        out[int(big["abs_r0"]) + int(dr) + rr, int(big["abs_c0"]) + int(dc) + cc] = int(small["color"])
    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_MAX_ATTEMPTS = 500
GEN_LAYOUTS = ("both", "rows", "cols")
GEN_PAIR_COUNTS = (2, 3)
GEN_TOTAL_LIMIT = 30
GEN_COLOR_POOL = tuple(range(1, 10))


def _sample_sizes(count: int, low: int, high: int, limit: int) -> list[int] | None:
    for _ in range(200):
        sizes = [random.randint(int(low), int(high)) for _ in range(int(count))]
        if sum(sizes) + max(0, count - 1) <= int(limit):
            return sizes
    return None


def _sizes_to_spans(sizes: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, size in enumerate(sizes):
        spans.append((int(cursor), int(cursor) + int(size) - 1))
        cursor += int(size)
        if i + 1 < len(sizes):
            cursor += 1
    return spans


def _layout_spans(mode: str, pair_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if mode == "both":
        row_count, col_count = (2, 2) if pair_count == 2 else random.choice(((2, 3), (3, 2)))
        row_sizes = _sample_sizes(row_count, 6, 9, GEN_TOTAL_LIMIT)
        col_sizes = _sample_sizes(col_count, 6, 9, GEN_TOTAL_LIMIT)
        if row_sizes is None or col_sizes is None:
            return None
        return _sizes_to_spans(row_sizes), _sizes_to_spans(col_sizes)
    if mode == "rows":
        row_sizes = _sample_sizes(2 * pair_count, 4, 6, GEN_TOTAL_LIMIT)
        if row_sizes is None:
            return None
        width = random.randint(10, 16)
        return _sizes_to_spans(row_sizes), [(0, int(width) - 1)]
    if mode == "cols":
        col_sizes = _sample_sizes(2 * pair_count, 4, 6, GEN_TOTAL_LIMIT)
        if col_sizes is None:
            return None
        height = random.randint(10, 16)
        return [(0, int(height) - 1)], _sizes_to_spans(col_sizes)
    raise AssertionError("unknown layout mode")


def _build_canvas(row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], sep_color: int) -> np.ndarray:
    h = int(row_spans[-1][1]) + 1
    w = int(col_spans[-1][1]) + 1
    grid = np.full((h, w), GEN_BG, dtype=int)
    for i in range(len(row_spans) - 1):
        grid[row_spans[i][1] + 1, :] = int(sep_color)
    for i in range(len(col_spans) - 1):
        grid[:, col_spans[i][1] + 1] = int(sep_color)
    return grid


def _panel_specs(row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]) -> list[dict]:
    panels: list[dict] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panels.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "h": int(r1 - r0 + 1),
                    "w": int(c1 - c0 + 1),
                }
            )
    return panels


def _grow_connected(area: int, h: int, w: int, start_mask: np.ndarray | None = None) -> np.ndarray | None:
    if start_mask is None:
        mask = np.zeros((h, w), dtype=bool)
        mask[random.randrange(h), random.randrange(w)] = True
    else:
        mask = start_mask.copy().astype(bool)
    while int(mask.sum()) < int(area):
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for nr, nc in _neighbors4(int(r), int(c), h, w):
                if not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True
    return mask


def _spanning_mask(h: int, w: int, area: int) -> np.ndarray | None:
    for _ in range(200):
        mask = _grow_connected(area, h, w)
        if mask is None:
            continue
        if (
            mask[0, :].any()
            and mask[-1, :].any()
            and mask[:, 0].any()
            and mask[:, -1].any()
        ):
            return mask
    return None


def _random_small_mask(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    h_choices = list(range(2, min(4, int(max_h)) + 1))
    w_choices = list(range(2, min(4, int(max_w)) + 1))
    for _ in range(200):
        h = random.choice(h_choices)
        w = random.choice(w_choices)
        min_area = max(3, max(h, w))
        max_area = min(5, h * w)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = _spanning_mask(h, w, area)
        if mask is None:
            continue
        if area == h * w and random.random() < 0.7:
            continue
        return mask
    return None


def _random_big_mask(small_mask: np.ndarray, max_h: int, max_w: int) -> np.ndarray | None:
    sh, sw = map(int, small_mask.shape)
    small_area = int(small_mask.sum())
    if max_h < sh or max_w < sw:
        return None
    for _ in range(300):
        temp_h = random.randint(sh, min(int(max_h), sh + 2))
        temp_w = random.randint(sw, min(int(max_w), sw + 2))
        max_area = min(temp_h * temp_w, small_area + 5)
        min_area = small_area + 2
        if min_area > max_area:
            continue
        off_r = random.randint(0, temp_h - sh)
        off_c = random.randint(0, temp_w - sw)
        seed = np.zeros((temp_h, temp_w), dtype=bool)
        seed[off_r : off_r + sh, off_c : off_c + sw][small_mask] = True
        area = random.randint(min_area, max_area)
        grown = _grow_connected(area, temp_h, temp_w, start_mask=seed)
        if grown is None:
            continue
        mask, _origin = _crop_mask(grown)
        if int(mask.sum()) <= small_area:
            continue
        if mask.shape[0] > int(max_h) or mask.shape[1] > int(max_w):
            continue
        if len(_embedding_positions(mask, small_mask)) != 1:
            continue
        return mask
    return None


def _place_in_panel(
    grid: np.ndarray,
    panel: dict,
    mask: np.ndarray,
    color: int,
) -> tuple[int, int]:
    h, w = map(int, mask.shape)
    max_r = int(panel["h"]) - h - 1
    max_c = int(panel["w"]) - w - 1
    assert max_r >= 1 and max_c >= 1, "panel must leave a border around the object"
    r = random.randint(1, max_r)
    c = random.randint(1, max_c)
    view = grid[int(panel["r0"]) + r : int(panel["r0"]) + r + h, int(panel["c0"]) + c : int(panel["c0"]) + c + w]
    view[mask] = int(color)
    return int(r), int(c)


def _all_matches(
    small_entries: list[dict],
    big_entries: list[dict],
) -> list[tuple[dict, dict, int, int]] | None:
    matches: list[tuple[dict, dict, int, int]] = []
    used_big_ids: set[int] = set()
    for small in small_entries:
        found: list[tuple[dict, int, int]] = []
        for big in big_entries:
            for dr, dc in _embedding_positions(big["mask"], small["mask"]):
                found.append((big, int(dr), int(dc)))
        if len(found) != 1:
            return None
        big, dr, dc = found[0]
        if int(big["slot"]) in used_big_ids:
            return None
        used_big_ids.add(int(big["slot"]))
        matches.append((small, big, dr, dc))
    if len(used_big_ids) != len(big_entries):
        return None
    return matches


def _render_expected(inp: np.ndarray, matches: list[tuple[dict, dict, int, int]]) -> np.ndarray:
    out = inp.copy()
    for small, big, dr, dc in matches:
        rr, cc = np.where(small["mask"])
        out[int(big["abs_r0"]) + int(dr) + rr, int(big["abs_c0"]) + int(dc) + cc] = int(small["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(GEN_PAIR_COUNTS)
    mode = random.choice(GEN_LAYOUTS)
    spans = _layout_spans(mode, pair_count)
    if spans is None:
        return None
    row_spans, col_spans = spans
    panels = _panel_specs(row_spans, col_spans)
    if len(panels) != 2 * pair_count:
        return None

    sep_color = random.choice(GEN_COLOR_POOL)
    object_colors = random.sample([c for c in GEN_COLOR_POOL if c != sep_color], 2 * pair_count)
    grid = _build_canvas(row_spans, col_spans, sep_color)

    slots = list(range(len(panels)))
    random.shuffle(slots)
    small_slots = slots[:pair_count]
    big_slots = slots[pair_count:]

    small_entries: list[dict] = []
    big_entries: list[dict] = []

    for i in range(pair_count):
        small_panel = panels[small_slots[i]]
        big_panel = panels[big_slots[i]]
        small_cap_h = int(small_panel["h"]) - 2
        small_cap_w = int(small_panel["w"]) - 2
        big_cap_h = int(big_panel["h"]) - 2
        big_cap_w = int(big_panel["w"]) - 2
        small_mask = _random_small_mask(small_cap_h, small_cap_w)
        if small_mask is None:
            return None
        big_mask = _random_big_mask(small_mask, big_cap_h, big_cap_w)
        if big_mask is None:
            return None
        small_entries.append(
            {
                "slot": int(small_slots[i]),
                "panel": small_panel,
                "mask": small_mask,
                "color": int(object_colors[i]),
                "area": int(small_mask.sum()),
            }
        )
        big_entries.append(
            {
                "slot": int(big_slots[i]),
                "panel": big_panel,
                "mask": big_mask,
                "color": int(object_colors[pair_count + i]),
                "area": int(big_mask.sum()),
            }
        )

    if max(int(entry["area"]) for entry in small_entries) >= min(int(entry["area"]) for entry in big_entries):
        return None

    matches = _all_matches(small_entries, big_entries)
    if matches is None:
        return None

    for entry in small_entries:
        rel_r, rel_c = _place_in_panel(grid, entry["panel"], entry["mask"], entry["color"])
        entry["abs_r0"] = int(entry["panel"]["r0"]) + int(rel_r)
        entry["abs_c0"] = int(entry["panel"]["c0"]) + int(rel_c)
    for entry in big_entries:
        rel_r, rel_c = _place_in_panel(grid, entry["panel"], entry["mask"], entry["color"])
        entry["abs_r0"] = int(entry["panel"]["r0"]) + int(rel_r)
        entry["abs_c0"] = int(entry["panel"]["c0"]) + int(rel_c)

    expected = _render_expected(grid, matches)
    if np.array_equal(expected, grid):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
