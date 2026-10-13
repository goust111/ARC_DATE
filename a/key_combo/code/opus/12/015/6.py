import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 6
CYCLE = (2, 8, 5)
ANCHOR_POOL = (1, 3, 4, 7, 9)
BODY_POOL = (1, 3, 4, 7, 9)
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left and top + h <= dst.shape[0] and left + w <= dst.shape[1]
    dst[top : top + h, left : left + w] = patch


def _patches_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    return not (ar + 2 < br or br + 2 < ar or ac + 2 < bc or bc + 2 < ac)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for coords in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        if not np.all(grid[r0, c0 : c1 + 1] == FRAME):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == FRAME):
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == FRAME):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == FRAME):
            continue
        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == FRAME):
            continue
        candidates.append((len(coords), (r0, c0, r1, c1)))
    assert candidates, "expected one magenta frame"
    candidates.sort(reverse=True)
    return candidates[0][1]


def _parse_grid(grid: np.ndarray):
    frame = _find_frame(grid)
    fr0, fc0, fr1, fc1 = frame
    sprites = []
    markers: dict[int, tuple[int, int]] = {}
    fg_mask = (grid != BG) & (grid != FRAME)
    for coords in _components(fg_mask):
        area = len(coords)
        colors = Counter(int(grid[r, c]) for r, c in coords)
        inside_frame = all(fr0 < r < fr1 and fc0 < c < fc1 for r, c in coords)
        if area == 1 and inside_frame:
            color = next(iter(colors))
            assert color not in markers, "marker colors must be unique"
            markers[color] = coords[0]
            continue
        box = _bbox(coords)
        patch = _crop(grid, box)
        assert patch.shape == (3, 3) and area == 9, "sprites must be solid 3x3 blocks"
        assert len(colors) == 2, "each sprite must have exactly two colors"
        anchor_colors = [color for color, count in colors.items() if count == 1]
        assert len(anchor_colors) == 1, "each sprite needs one singleton anchor color"
        anchor_color = int(anchor_colors[0])
        anchor_pos = np.argwhere(patch == anchor_color)
        assert anchor_pos.shape == (1, 2), "anchor color must appear once"
        ar, ac = map(int, anchor_pos[0])
        sprites.append(
            {
                "bbox": box,
                "top_left": (box[0], box[1]),
                "patch": patch,
                "anchor_color": anchor_color,
                "anchor_local": (ar, ac),
                "anchor_global": (box[0] + ar, box[1] + ac),
            }
        )
    anchor_colors = {sprite["anchor_color"] for sprite in sprites}
    assert len(anchor_colors) == len(sprites) >= 3, "sprites must have distinct anchors"
    assert set(markers) == anchor_colors, "frame markers must match sprite anchors"
    return frame, sprites, markers


def _recolor_patch(anchor_color: int, anchor_local: tuple[int, int], body_color: int) -> np.ndarray:
    patch = np.full((3, 3), int(body_color), dtype=int)
    ar, ac = anchor_local
    patch[ar, ac] = int(anchor_color)
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each anchored sprite by scan order, then stamp the recolored sprites into the matching frame markers.
    
    Input structure:
    - A black background contains several separated 3x3 source sprites and one magenta bordered frame.
    - Every source sprite is a solid 3x3 two-color block with exactly one singleton anchor color.
    - Inside the frame, singleton marker pixels use those anchor colors to indicate stamp destinations.
    
    Rule:
    - Identify each sprite's anchor as its unique least-frequent color, then sort sprites by the anchor pixel's row-major position.
    - Recolor only the non-anchor cells of the 1st, 2nd, 3rd, ... sprites with the repeating cycle red, teal, grey while keeping anchors unchanged.
    - Clear the frame interior to black and stamp each recolored sprite so its anchor lands on the same-colored frame marker; later sprites in scan order overwrite earlier overlaps.
    
    Key insight:
    - The anchor color is the only stable cue across the whole chain: it determines sort order reference, survives recoloring, and matches each stamped destination.
    """
    g = np.asarray(grid, dtype=int)
    frame, sprites, markers = _parse_grid(g)
    fr0, fc0, fr1, fc1 = frame
    ordered = sorted(sprites, key=lambda sprite: sprite["anchor_global"])

    out = g.copy()
    recolored: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}
    for i, sprite in enumerate(ordered):
        patch = _recolor_patch(sprite["anchor_color"], sprite["anchor_local"], CYCLE[i % len(CYCLE)])
        r0, c0 = sprite["top_left"]
        out[r0 : r0 + 3, c0 : c0 + 3] = patch
        recolored[sprite["anchor_color"]] = (patch, sprite["anchor_local"])

    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = BG
    for sprite in ordered:
        patch, (ar, ac) = recolored[sprite["anchor_color"]]
        mr, mc = markers[sprite["anchor_color"]]
        top = int(mr - ar)
        left = int(mc - ac)
        assert fr0 < top and fc0 < left and top + 2 < fr1 and left + 2 < fc1
        _paste(out, patch, top, left)

    out[fr0, fc0 : fc1 + 1] = FRAME
    out[fr1, fc0 : fc1 + 1] = FRAME
    out[fr0 : fr1 + 1, fc0] = FRAME
    out[fr0 : fr1 + 1, fc1] = FRAME
    return out


def _build_input_from_blueprint(spec: dict) -> np.ndarray:
    grid = np.full((spec["height"], spec["width"]), BG, dtype=int)
    fr0, fc0, fr1, fc1 = spec["frame"]
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME

    for sprite in spec["sprites"]:
        sr, sc = sprite["source_top_left"]
        patch = np.full((3, 3), int(sprite["body_color"]), dtype=int)
        ar, ac = sprite["anchor_local"]
        patch[ar, ac] = int(sprite["anchor_color"])
        grid[sr : sr + 3, sc : sc + 3] = patch

        tr, tc = sprite["target_top_left"]
        mr = fr0 + 1 + tr + ar
        mc = fc0 + 1 + tc + ac
        grid[mr, mc] = int(sprite["anchor_color"])

    return grid


def _build_expected_output(spec: dict) -> np.ndarray:
    out = np.full((spec["height"], spec["width"]), BG, dtype=int)
    fr0, fc0, fr1, fc1 = spec["frame"]
    out[fr0, fc0 : fc1 + 1] = FRAME
    out[fr1, fc0 : fc1 + 1] = FRAME
    out[fr0 : fr1 + 1, fc0] = FRAME
    out[fr0 : fr1 + 1, fc1] = FRAME

    ordered = sorted(
        spec["sprites"],
        key=lambda sprite: (
            sprite["source_top_left"][0] + sprite["anchor_local"][0],
            sprite["source_top_left"][1] + sprite["anchor_local"][1],
        ),
    )
    for i, sprite in enumerate(ordered):
        recolored = _recolor_patch(sprite["anchor_color"], sprite["anchor_local"], CYCLE[i % len(CYCLE)])
        sr, sc = sprite["source_top_left"]
        out[sr : sr + 3, sc : sc + 3] = recolored

    for i, sprite in enumerate(ordered):
        recolored = _recolor_patch(sprite["anchor_color"], sprite["anchor_local"], CYCLE[i % len(CYCLE)])
        tr, tc = sprite["target_top_left"]
        top = fr0 + 1 + tr
        left = fc0 + 1 + tc
        out[top : top + 3, left : left + 3] = recolored
    return out


def _choose_source_slots(height: int, count: int, want_row_tie: bool):
    row_slots = list(range(1, height - 2, 4))
    col_slots = [1, 5]
    slots = [(r, c) for r in row_slots for c in col_slots]
    if len(slots) < count:
        return None
    if want_row_tie and count >= 2 and row_slots:
        tie_row = random.choice(row_slots)
        chosen = [(tie_row, col_slots[0]), (tie_row, col_slots[1])]
        remaining = [slot for slot in slots if slot not in chosen]
        chosen.extend(random.sample(remaining, count - 2))
        random.shuffle(chosen)
        return chosen
    return random.sample(slots, count)


def _choose_body_colors(anchor_colors: list[int]) -> list[int]:
    if random.random() < 0.6:
        repeated = random.choice(BODY_POOL)
        colors = [repeated] * len(anchor_colors)
        for i, anchor in enumerate(anchor_colors):
            if colors[i] == anchor:
                choices = [color for color in BODY_POOL if color != anchor]
                colors[i] = random.choice(choices)
        return colors
    colors = []
    for anchor in anchor_colors:
        choices = [color for color in BODY_POOL if color != anchor]
        colors.append(random.choice(choices))
    return colors


def _choose_anchor_locs(source_slots: list[tuple[int, int]], want_row_tie: bool):
    locs = [(r, c) for r in range(3) for c in range(3)]
    chosen = [random.choice(locs) for _ in source_slots]
    if want_row_tie:
        row_groups: dict[int, list[int]] = {}
        for idx, (r0, _c0) in enumerate(source_slots):
            row_groups.setdefault(r0, []).append(idx)
        tie_group = next((indices for indices in row_groups.values() if len(indices) >= 2), None)
        if tie_group is not None:
            anchor_row = random.randrange(3)
            for idx in tie_group[:2]:
                chosen[idx] = (anchor_row, random.randrange(3))
            if chosen[tie_group[0]] == chosen[tie_group[1]]:
                c = (chosen[tie_group[1]][1] + 1) % 3
                chosen[tie_group[1]] = (anchor_row, c)
    return chosen


def _marker_coords(frame: tuple[int, int, int, int], target_tops, anchor_locs):
    fr0, fc0, _fr1, _fc1 = frame
    return [
        (fr0 + 1 + top_r + ar, fc0 + 1 + top_c + ac)
        for (top_r, top_c), (ar, ac) in zip(target_tops, anchor_locs)
    ]


def _markers_are_isolated(markers: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(markers):
        for r2, c2 in markers[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) <= 1:
                return False
    return len(set(markers)) == len(markers)


def _choose_target_tops(inner_h: int, inner_w: int, anchor_locs, want_overlap: bool):
    all_tops = [(r, c) for r in range(inner_h - 2) for c in range(inner_w - 2)]
    if len(all_tops) < len(anchor_locs):
        return None
    for _ in range(300):
        tops = random.sample(all_tops, len(anchor_locs))
        markers = [(top_r + ar, top_c + ac) for (top_r, top_c), (ar, ac) in zip(tops, anchor_locs)]
        if not _markers_are_isolated(markers):
            continue
        overlaps = any(_patches_overlap(a, b) for i, a in enumerate(tops) for b in tops[i + 1 :])
        if overlaps != want_overlap:
            continue
        return tops
    return None


def _validate_blueprint(spec: dict, grid: np.ndarray, expected: np.ndarray) -> bool:
    if grid.shape != expected.shape:
        return False
    if np.array_equal(grid, expected):
        return False
    if max(grid.shape) > 30:
        return False

    frame, sprites, markers = _parse_grid(grid)
    if len(sprites) != len(spec["sprites"]) or len(markers) != len(spec["sprites"]):
        return False

    ordered = sorted(sprites, key=lambda sprite: sprite["anchor_global"])
    if [sprite["anchor_color"] for sprite in ordered] != spec["ordered_anchor_colors"]:
        return False

    overlaps = any(
        _patches_overlap(a["target_top_left"], b["target_top_left"])
        for i, a in enumerate(spec["sprites"])
        for b in spec["sprites"][i + 1 :]
    )
    if overlaps != spec["want_overlap"]:
        return False

    tie_rows = [sprite["source_top_left"][0] + sprite["anchor_local"][0] for sprite in spec["sprites"]]
    has_row_tie = len(tie_rows) != len(set(tie_rows))
    if has_row_tie != spec["want_row_tie"]:
        return False

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    fr0, fc0, fr1, fc1 = frame
    if not np.all(actual[fr0, fc0 : fc1 + 1] == FRAME):
        return False
    if not np.all(actual[fr1, fc0 : fc1 + 1] == FRAME):
        return False
    if not np.all(actual[fr0 : fr1 + 1, fc0] == FRAME):
        return False
    if not np.all(actual[fr0 : fr1 + 1, fc1] == FRAME):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choices([3, 4, 5], weights=[2, 4, 4], k=1)[0]
    want_row_tie = count >= 2 and random.random() < 0.5
    want_overlap = count >= 2 and random.random() < 0.55

    height = random.randint(18, 24)
    inner_h_min = 7 if want_overlap else 8
    inner_w_min = 7 if want_overlap else 8
    inner_h = random.randint(inner_h_min, 9)
    inner_w = random.randint(inner_w_min, 9)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    source_width = random.choice([8, 9, 10])
    gap = random.randint(2, 3)
    width = 1 + source_width + gap + frame_w + 1
    if width > 30:
        return None

    frame_r0 = random.randint(1, height - frame_h - 1)
    frame_c0 = 1 + source_width + gap
    frame = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

    source_slots = _choose_source_slots(height, count, want_row_tie)
    if source_slots is None:
        return None
    anchor_colors = random.sample(list(ANCHOR_POOL), count)
    body_colors = _choose_body_colors(anchor_colors)
    anchor_locs = _choose_anchor_locs(source_slots, want_row_tie)
    target_tops = _choose_target_tops(inner_h, inner_w, anchor_locs, want_overlap)
    if target_tops is None:
        return None

    sprites = []
    for idx in range(count):
        sprites.append(
            {
                "source_top_left": source_slots[idx],
                "target_top_left": target_tops[idx],
                "anchor_local": anchor_locs[idx],
                "anchor_color": anchor_colors[idx],
                "body_color": body_colors[idx],
            }
        )

    ordered = sorted(
        sprites,
        key=lambda sprite: (
            sprite["source_top_left"][0] + sprite["anchor_local"][0],
            sprite["source_top_left"][1] + sprite["anchor_local"][1],
        ),
    )
    spec = {
        "height": height,
        "width": width,
        "frame": frame,
        "sprites": sprites,
        "want_overlap": want_overlap,
        "want_row_tie": want_row_tie,
        "ordered_anchor_colors": [sprite["anchor_color"] for sprite in ordered],
    }
    grid = _build_input_from_blueprint(spec)
    expected = _build_expected_output(spec)
    if not _validate_blueprint(spec, grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
