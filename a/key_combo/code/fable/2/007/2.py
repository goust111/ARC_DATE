from __future__ import annotations

import random
from collections import Counter

import numpy as np


SOURCE_BG = 0
TARGET_BG = 1
RED = 2
DIVIDER = 4
REMNANT = 5
SPRITE_COLORS = (3, 6, 7, 8, 9)

PANEL_H_RANGE = (12, 14)
PANEL_W_RANGE = (12, 14)
SPRITE_H_RANGE = (4, 7)
SPRITE_W_RANGE = (4, 7)
SPRITE_COUNT_CHOICES = (3, 4)
REMNANT_COUNT_CHOICES = (2, 3)
PLACEMENT_BORDER = 1
PLACEMENT_PADDING = 1
GEN_ATTEMPTS = 500
SHAPE_ATTEMPTS = 80


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    monochromatic: bool,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c]) if monochromatic else None
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    if monochromatic:
                        if int(grid[nr, nc]) != color:
                            continue
                    else:
                        if int(grid[nr, nc]) == bg:
                            continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            r0, c0 = coord_arr.min(axis=0)
            r1, c1 = coord_arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _sprite_records(source: np.ndarray) -> tuple[int, list[dict]]:
    source_bg = _get_bg(source)
    components = _find_components(source, bg=source_bg, monochromatic=False)
    if not components:
        raise AssertionError("source panel must contain sprites")
    records: list[dict] = []
    for comp in components:
        patch = _crop(source, comp["bbox"])
        records.append({"patch": patch, "bbox": comp["bbox"]})
    return source_bg, records


def _solve_panels(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    source_bg, sprites = _sprite_records(source)
    target_bg = _get_bg(target)
    target_colors = sorted(set(map(int, np.unique(target))) - {target_bg})
    if len(target_colors) != 1:
        raise AssertionError("target panel must contain exactly one remnant color")
    remnant_color = target_colors[0]
    remnants = _find_components(target, bg=target_bg, monochromatic=True)
    if not remnants or any(int(rem["color"]) != remnant_color for rem in remnants):
        raise AssertionError("target foreground must be gray remnant components")
    remnants.sort(key=lambda item: item["bbox"])

    matches: list[dict] = []
    for remnant in remnants:
        rem_box = remnant["bbox"]
        rem_mask = _crop(target == remnant_color, rem_box)
        candidates: list[dict] = []
        for sprite_index, sprite in enumerate(sprites):
            for mirror in (False, True):
                patch = np.fliplr(sprite["patch"]).copy() if mirror else sprite["patch"].copy()
                mask = patch != source_bg
                sprite_h, sprite_w = mask.shape
                for prefix_h in range(1, sprite_h):
                    prefix_full = mask[:prefix_h, :]
                    prefix_box = _bbox(prefix_full)
                    if prefix_box is None:
                        continue
                    prefix_mask = _crop(prefix_full, prefix_box)
                    if prefix_mask.shape != rem_mask.shape:
                        continue
                    if not np.array_equal(prefix_mask, rem_mask):
                        continue
                    top_r = rem_box[0] - prefix_box[0]
                    top_c = rem_box[1] - prefix_box[1]
                    if top_r < 0 or top_c < 0:
                        continue
                    if top_r + sprite_h > target.shape[0] or top_c + sprite_w > target.shape[1]:
                        continue
                    candidates.append(
                        {
                            "sprite_index": sprite_index,
                            "patch": patch,
                            "mirror": mirror,
                            "prefix_h": prefix_h,
                            "top": (top_r, top_c),
                            "rem_box": rem_box,
                        }
                    )
        if len(candidates) != 1:
            raise AssertionError("every remnant must admit exactly one source/truncation/mirror match")
        matches.append(candidates[0])

    out = target.copy()
    written: dict[tuple[int, int], int] = {}
    preserved_cells: set[tuple[int, int]] = set()
    input_fg = {tuple(map(int, cell)) for cell in np.argwhere(target != target_bg)}

    for match in matches:
        patch = match["patch"]
        prefix_h = int(match["prefix_h"])
        top_r, top_c = match["top"]
        for r, c in np.argwhere(patch != source_bg):
            rr = top_r + int(r)
            cc = top_c + int(c)
            if int(r) < prefix_h:
                if int(target[rr, cc]) != remnant_color:
                    raise AssertionError("preserved cells must coincide with the gray remnant")
                color = int(patch[r, c])
                preserved_cells.add((rr, cc))
            else:
                if int(target[rr, cc]) != target_bg:
                    raise AssertionError("inferred cells must occupy empty target background")
                color = RED
            prev = written.get((rr, cc))
            if prev is not None and prev != color:
                raise AssertionError("reconstructed sprites must not conflict")
            written[(rr, cc)] = color

    if preserved_cells != input_fg:
        raise AssertionError("all and only remnant cells must be restored from true sprite colors")

    for (r, c), color in written.items():
        out[r, c] = color
    return out, matches


def _separator_splits(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    splits: list[dict] = []
    for r in range(1, h - 1):
        color = int(g[r, 0])
        if np.all(g[r, :] == color) and not np.any(g[:r, :] == color) and not np.any(g[r + 1 :, :] == color):
            splits.append(
                {
                    "axis": "horizontal",
                    "index": r,
                    "panel_a": g[:r, :],
                    "panel_b": g[r + 1 :, :],
                    "box_a": (0, 0, r - 1, w - 1),
                    "box_b": (r + 1, 0, h - 1, w - 1),
                }
            )
    for c in range(1, w - 1):
        color = int(g[0, c])
        if np.all(g[:, c] == color) and not np.any(g[:, :c] == color) and not np.any(g[:, c + 1 :] == color):
            splits.append(
                {
                    "axis": "vertical",
                    "index": c,
                    "panel_a": g[:, :c],
                    "panel_b": g[:, c + 1 :],
                    "box_a": (0, 0, h - 1, c - 1),
                    "box_b": (0, c + 1, h - 1, w - 1),
                }
            )
    return splits


def _write_panel(dst: np.ndarray, box: tuple[int, int, int, int], panel: np.ndarray) -> None:
    r0, c0, r1, c1 = box
    assert panel.shape == (r1 - r0 + 1, c1 - c0 + 1)
    dst[r0 : r1 + 1, c0 : c1 + 1] = panel


def _enumerate_solutions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    solutions: list[dict] = []
    for split in _separator_splits(g):
        for source_key, target_key in (("a", "b"), ("b", "a")):
            source_panel = split[f"panel_{source_key}"]
            target_panel = split[f"panel_{target_key}"]
            try:
                rebuilt_target, matches = _solve_panels(source_panel, target_panel)
            except AssertionError:
                continue
            out = g.copy()
            _write_panel(out, split[f"box_{target_key}"], rebuilt_target)
            solutions.append(
                {
                    "output": out,
                    "axis": split["axis"],
                    "source_side": source_key,
                    "match_count": len(matches),
                    "matches": matches,
                }
            )
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each gray remnant into its unique source sprite copy, coloring only the inferred missing rows red.

    Input structure:
    - The grid is split by a single full-width or full-height divider line into two rectangular panels.
    - One panel contains several separated multicolor source sprites on its own background.
    - The other panel contains separated gray remnants on a different background; each remnant is a source sprite with one or more bottom rows removed, possibly after a left-right mirror, and with all surviving cells recolored gray.

    Rule:
    - Infer the divider orientation and which panel is the source versus the remnant panel.
    - For every gray remnant, enumerate source sprite, truncation height, and mirror state, and keep the unique candidate whose top-prefix shape matches the remnant mask.
    - Reconstruct the full matched sprite at the remnant location: surviving remnant cells recover their original mirrored-or-not source colors, while every cell that belongs only to the removed bottom rows is added back in red.
    - Leave the source panel and the divider unchanged.

    Key insight:
    - The gray object is not a complete symbol to recolor; it is a shape-only prefix of a hidden full sprite, so both the missing row count and the mirror state must be inferred from unique prefix-shape agreement before colors can be restored.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.size == 0:
        raise AssertionError("expected a non-empty 2D grid")
    solutions = _enumerate_solutions(g)
    if len(solutions) != 1:
        raise AssertionError("expected exactly one valid source/target interpretation")
    return solutions[0]["output"]


def _shape_mask(height: int, width: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        rows: list[tuple[int, int]] = [(0, width - 1)]
        left, right = 0, width - 1
        changed = False
        for _row in range(1, height):
            candidates: list[tuple[int, int]] = []
            for dl in (-1, 0, 1):
                new_left = min(max(left + dl, 0), width - 1)
                for dr in (-1, 0, 1):
                    new_right = min(max(right + dr, new_left), width - 1)
                    if new_left > right or new_right < left:
                        continue
                    candidates.append((new_left, new_right))
            if not candidates:
                break
            new_left, new_right = random.choice(candidates)
            changed |= (new_left, new_right) != (left, right)
            rows.append((new_left, new_right))
            left, right = new_left, new_right
        if len(rows) != height or not changed:
            continue
        mask = np.zeros((height, width), dtype=bool)
        for r, (c0, c1) in enumerate(rows):
            mask[r, c0 : c1 + 1] = True
        area = int(mask.sum())
        if area < height * 2 or area > int(round(height * width * 0.88)):
            continue
        if np.all(mask):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        if len({tuple(row) for row in mask.astype(int)}) < 2:
            continue
        return mask
    return None


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    color_count = random.choice((2, 3))
    colors = random.sample(SPRITE_COLORS, color_count)
    phase = random.randrange(color_count)
    patch = np.full(mask.shape, SOURCE_BG, dtype=int)
    for r, c in np.argwhere(mask):
        patch[int(r), int(c)] = colors[(phase + int(c) + 2 * int(r)) % color_count]
    used = set(map(int, patch[mask]))
    if len(used) < 2:
        coords = list(map(tuple, np.argwhere(mask)))
        for idx, (r, c) in enumerate(coords):
            patch[r, c] = colors[idx % len(colors)]
    return patch


def _valid_combo_pool(patches: list[np.ndarray]) -> list[dict]:
    combos: list[dict] = []
    counts: Counter = Counter()
    for source_index, patch in enumerate(patches):
        for mirror in (False, True):
            oriented = np.fliplr(patch).copy() if mirror else patch.copy()
            mask = oriented != SOURCE_BG
            height = oriented.shape[0]
            for prefix_h in range(1, height):
                prefix_mask = mask[:prefix_h, :]
                box = _bbox(prefix_mask)
                if box is None:
                    continue
                signature = _mask_signature(_crop(prefix_mask, box))
                combo = {
                    "source_index": source_index,
                    "mirror": mirror,
                    "prefix_h": prefix_h,
                    "patch": oriented,
                    "signature": signature,
                }
                combos.append(combo)
                counts[signature] += 1
    return [combo for combo in combos if counts[combo["signature"]] == 1]


def _pick_combos(pool: list[dict]) -> list[dict] | None:
    if len(pool) < 2:
        return None
    target_count = random.choice(REMNANT_COUNT_CHOICES)
    target_count = min(target_count, len(pool))
    chosen: list[dict] = []
    used_sigs: set[tuple[tuple[int, int], bytes]] = set()
    used_sources: set[int] = set()

    def pick(candidates: list[dict], *, prefer_new_source: bool) -> bool:
        options = [c for c in candidates if c["signature"] not in used_sigs]
        if prefer_new_source:
            fresh = [c for c in options if c["source_index"] not in used_sources]
            if fresh:
                options = fresh
        if not options:
            return False
        combo = random.choice(options)
        chosen.append(combo)
        used_sigs.add(combo["signature"])
        used_sources.add(int(combo["source_index"]))
        return True

    mirrored = [combo for combo in pool if combo["mirror"]]
    direct = [combo for combo in pool if not combo["mirror"]]
    deep = [combo for combo in pool if combo["prefix_h"] <= combo["patch"].shape[0] - 2]

    if target_count >= 2 and mirrored and direct and random.random() < 0.7:
        pick(mirrored, prefer_new_source=True)
        if len(chosen) < target_count:
            pick(direct, prefer_new_source=True)
    if len(chosen) < target_count and deep and random.random() < 0.6:
        pick(deep, prefer_new_source=True)

    while len(chosen) < target_count:
        if pick(pool, prefer_new_source=True):
            continue
        if pick(pool, prefer_new_source=False):
            continue
        return None

    source_variety = {int(combo["source_index"]) for combo in pool}
    if target_count >= 2 and len(source_variety) >= 2 and len({int(combo["source_index"]) for combo in chosen}) < 2:
        return None
    return chosen


def _random_box_positions(
    panel_shape: tuple[int, int],
    boxes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    panel_h, panel_w = panel_shape
    order = sorted(range(len(boxes)), key=lambda idx: boxes[idx][0] * boxes[idx][1], reverse=True)
    reserved = np.zeros(panel_shape, dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(boxes)

    for idx in order:
        box_h, box_w = boxes[idx]
        candidates: list[tuple[int, int]] = []
        for r0 in range(PLACEMENT_BORDER, panel_h - box_h - PLACEMENT_BORDER + 1):
            for c0 in range(PLACEMENT_BORDER, panel_w - box_w - PLACEMENT_BORDER + 1):
                if reserved[r0 : r0 + box_h, c0 : c0 + box_w].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        rr0 = max(0, r0 - PLACEMENT_PADDING)
        cc0 = max(0, c0 - PLACEMENT_PADDING)
        rr1 = min(panel_h, r0 + box_h + PLACEMENT_PADDING)
        cc1 = min(panel_w, c0 + box_w + PLACEMENT_PADDING)
        reserved[rr0:rr1, cc0:cc1] = True

    return [placement for placement in placements if placement is not None]


def _compose_full(
    source_panel: np.ndarray,
    target_panel: np.ndarray,
    *,
    axis: str,
    source_first: bool,
) -> np.ndarray:
    if axis == "vertical":
        if source_first:
            return np.hstack(
                [
                    source_panel,
                    np.full((source_panel.shape[0], 1), DIVIDER, dtype=int),
                    target_panel,
                ]
            )
        return np.hstack(
            [
                target_panel,
                np.full((source_panel.shape[0], 1), DIVIDER, dtype=int),
                source_panel,
            ]
        )
    if source_first:
        return np.vstack(
            [
                source_panel,
                np.full((1, source_panel.shape[1]), DIVIDER, dtype=int),
                target_panel,
            ]
        )
    return np.vstack(
        [
            target_panel,
            np.full((1, source_panel.shape[1]), DIVIDER, dtype=int),
            source_panel,
        ]
    )


def _build_panels() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    source_panel = np.full((panel_h, panel_w), SOURCE_BG, dtype=int)
    target_input = np.full((panel_h, panel_w), TARGET_BG, dtype=int)
    target_output = np.full((panel_h, panel_w), TARGET_BG, dtype=int)

    max_sprites = 4 if panel_h >= 13 and panel_w >= 13 else 3
    sprite_count = random.choice([count for count in SPRITE_COUNT_CHOICES if count <= max_sprites])

    patches: list[np.ndarray] = []
    for _ in range(sprite_count):
        height = random.randint(*SPRITE_H_RANGE)
        width = random.randint(*SPRITE_W_RANGE)
        mask = _shape_mask(height, width)
        if mask is None:
            return None
        patches.append(_colorize_mask(mask))

    source_positions = _random_box_positions((panel_h, panel_w), [patch.shape for patch in patches])
    if source_positions is None:
        return None
    for patch, (r0, c0) in zip(patches, source_positions):
        h, w = patch.shape
        source_panel[r0 : r0 + h, c0 : c0 + w][patch != SOURCE_BG] = patch[patch != SOURCE_BG]

    combo_pool = _valid_combo_pool(patches)
    chosen = _pick_combos(combo_pool)
    if chosen is None:
        return None

    target_positions = _random_box_positions((panel_h, panel_w), [combo["patch"].shape for combo in chosen])
    if target_positions is None:
        return None

    for combo, (r0, c0) in zip(chosen, target_positions):
        patch = combo["patch"]
        prefix_h = int(combo["prefix_h"])
        for r, c in np.argwhere(patch != SOURCE_BG):
            rr = r0 + int(r)
            cc = c0 + int(c)
            if int(r) < prefix_h:
                target_input[rr, cc] = REMNANT
                target_output[rr, cc] = int(patch[r, c])
            else:
                target_output[rr, cc] = RED

    return source_panel, target_input, target_output


def _generated_pair() -> tuple[np.ndarray, np.ndarray] | None:
    panels = _build_panels()
    if panels is None:
        return None
    source_panel, target_input, target_output = panels
    axis = random.choice(("vertical", "horizontal"))
    source_first = bool(random.getrandbits(1))
    inp = _compose_full(source_panel, target_input, axis=axis, source_first=source_first)
    out = _compose_full(source_panel, target_output, axis=axis, source_first=source_first)
    solutions = _enumerate_solutions(inp)
    if len(solutions) != 1:
        return None
    if not np.array_equal(solutions[0]["output"], out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair = _generated_pair()
        if pair is None:
            continue
        inp, _out = pair
        return inp
    raise RuntimeError("failed to generate a valid residual-reconstruction input")
