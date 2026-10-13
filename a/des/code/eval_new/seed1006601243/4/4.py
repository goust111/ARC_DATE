import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR_TO_STEP = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_TO_ROT_K = {
    "up": 0,
    "right": 3,
    "down": 2,
    "left": 1,
}
ORTHOGONAL = {
    "up": (0, 1),
    "right": (1, 0),
    "down": (0, 1),
    "left": (1, 0),
}
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _separator_row(g: np.ndarray) -> tuple[int, int]:
    rows = [
        r
        for r in range(g.shape[0])
        if int(g[r, 0]) != BG and np.all(g[r] == g[r, 0])
    ]
    assert len(rows) == 1, "expected one solid non-black separator row"
    return int(rows[0]), int(g[rows[0], 0])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(h, w, nr, nc):
                        continue
                    if not mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _candidate_tail_cells(pattern_r0: int, pattern_c0: int, direction: str) -> list[tuple[int, int]]:
    if direction == "up":
        return [(pattern_r0 - 1, pattern_c0 + dc) for dc in range(3)]
    if direction == "down":
        return [(pattern_r0 + 3, pattern_c0 + dc) for dc in range(3)]
    if direction == "left":
        return [(pattern_r0 + dr, pattern_c0 - 1) for dr in range(3)]
    return [(pattern_r0 + dr, pattern_c0 + 3) for dr in range(3)]


def _trace_arrow(
    top: np.ndarray,
    comp_mask: np.ndarray,
    pattern_r0: int,
    pattern_c0: int,
    direction: str,
    tail: tuple[int, int],
) -> dict | None:
    h, w = top.shape
    tr, tc = tail
    if not _in_bounds(h, w, tr, tc) or not comp_mask[tr, tc]:
        return None

    color = int(top[tr, tc])
    if color in (BG, SEP):
        return None

    dr, dc = DIR_TO_STEP[direction]
    line: list[tuple[int, int]] = []
    rr, cc = tr, tc
    while _in_bounds(h, w, rr, cc) and int(top[rr, cc]) == color:
        line.append((rr, cc))
        rr += dr
        cc += dc
    if len(line) < 2:
        return None

    tip_r, tip_c = line[-1]
    ortho = ORTHOGONAL[direction]
    a = (tip_r + ortho[0], tip_c + ortho[1])
    b = (tip_r - ortho[0], tip_c - ortho[1])
    if not (_in_bounds(h, w, *a) and _in_bounds(h, w, *b)):
        return None
    if int(top[a]) != color or int(top[b]) != color:
        return None

    pattern = top[pattern_r0 : pattern_r0 + 3, pattern_c0 : pattern_c0 + 3].copy()
    return {
        "arrow_color": color,
        "direction": direction,
        "length": len(line),
        "pattern": pattern,
    }


def _parse_stamp(top: np.ndarray, comp: np.ndarray) -> dict:
    h, w = top.shape
    comp_mask = np.zeros((h, w), dtype=bool)
    comp_mask[comp[:, 0], comp[:, 1]] = True

    seen: set[tuple[int, int, str, int, int]] = set()
    candidates: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if not comp_mask[r0 : r0 + 3, c0 : c0 + 3].all():
                continue
            for direction in ("up", "right", "down", "left"):
                for tail in _candidate_tail_cells(r0, c0, direction):
                    parsed = _trace_arrow(top, comp_mask, r0, c0, direction, tail)
                    if parsed is None:
                        continue
                    key = (
                        r0,
                        c0,
                        direction,
                        parsed["arrow_color"],
                        parsed["length"],
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(parsed)

    assert len(candidates) == 1, "expected one pattern-arrow parse per stamp"
    return candidates[0]


def _rotate_pattern(pattern: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(pattern, DIR_TO_ROT_K[direction]).copy()


def _apply_stamps(
    canvas_shape: tuple[int, int],
    stamps: list[dict],
    anchors: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = np.full(canvas_shape, BG, dtype=int)
    for stamp in sorted(stamps, key=lambda item: item["length"]):
        ar, ac = anchors[stamp["arrow_color"]]
        assert 1 <= ar < canvas_shape[0] - 1
        assert 1 <= ac < canvas_shape[1] - 1
        patch = _rotate_pattern(stamp["pattern"], stamp["direction"])
        out[ar - 1 : ar + 2, ac - 1 : ac + 2] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp each 3x3 tail pattern onto the lower canvas, rotating by arrow direction and overwriting by shaft length.

    Input structure:
    - Black background with one full-width grey separator row splitting the grid into an instruction panel and a canvas.
    - Above the separator, each connected stamp object contains a 3x3 multicolor pattern attached to the tail of a monochrome arrow.
    - Below the separator, each arrow color appears once as a single-pixel anchor on an otherwise black canvas.

    Rule:
    - Parse every upper stamp to recover its arrow color, pointing direction, shaft length, and attached 3x3 pattern.
    - Match each stamp to the lower anchor of the same arrow color.
    - Place the pattern centered on that anchor, using the rotations up=0, right=90 clockwise, down=180, left=270 clockwise.
    - Apply stamps from shorter shafts to longer shafts so later placements overwrite earlier overlap.

    Key insight:
    - The arrow encodes two independent instructions at once: its color selects the destination anchor, while its direction and shaft length control rotation and overwrite priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    sep_r, sep_color = _separator_row(g)

    top = g[:sep_r]
    bottom = g[sep_r + 1 :]
    fg_top = (top != BG) & (top != sep_color)
    comps = _connected_components(fg_top)
    assert comps, "expected at least one upper stamp"

    stamps = [_parse_stamp(top, comp) for comp in comps]

    anchors: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere(bottom != BG):
        color = int(bottom[r, c])
        assert color not in anchors, "anchor colors must be unique"
        anchors[color] = (int(r), int(c))
    assert len(anchors) == len(stamps), "stamp-anchor count mismatch"

    return _apply_stamps(bottom.shape, stamps, anchors)


def _random_pattern(pattern_colors: list[int]) -> np.ndarray:
    for _ in range(200):
        k = random.randint(2, min(4, len(pattern_colors)))
        colors = random.sample(pattern_colors, k=k)
        values = [random.choice(colors) for _ in range(9)]
        if len(set(values)) < 2:
            continue
        pattern = np.array(values, dtype=int).reshape(3, 3)
        rotations = {tuple(np.rot90(pattern, k).reshape(-1)) for k in range(4)}
        if len(rotations) == 4:
            return pattern
    raise RuntimeError("failed to sample a non-symmetric pattern")


def _render_stamp(spec: dict) -> np.ndarray:
    direction = spec["direction"]
    length = spec["length"]
    color = spec["arrow_color"]
    pattern = spec["pattern"]
    attach = spec["attach"]

    if direction in ("left", "right"):
        top_pad = 1 if attach == 0 else 0
        bottom_pad = 1 if attach == 2 else 0
        h = 3 + top_pad + bottom_pad
        w = 3 + length
        sprite = np.full((h, w), BG, dtype=int)
        pattern_r0 = top_pad
        shaft_r = pattern_r0 + attach
        if direction == "right":
            sprite[pattern_r0 : pattern_r0 + 3, :3] = pattern
            sprite[shaft_r, 3 : 3 + length] = color
            tip_c = 3 + length - 1
        else:
            sprite[pattern_r0 : pattern_r0 + 3, length : length + 3] = pattern
            sprite[shaft_r, :length] = color
            tip_c = 0
        sprite[shaft_r - 1, tip_c] = color
        sprite[shaft_r + 1, tip_c] = color
        return sprite

    left_pad = 1 if attach == 0 else 0
    right_pad = 1 if attach == 2 else 0
    h = 3 + length
    w = 3 + left_pad + right_pad
    sprite = np.full((h, w), BG, dtype=int)
    pattern_c0 = left_pad
    shaft_c = pattern_c0 + attach
    if direction == "down":
        sprite[:3, pattern_c0 : pattern_c0 + 3] = pattern
        sprite[3 : 3 + length, shaft_c] = color
        tip_r = 3 + length - 1
    else:
        sprite[length : length + 3, pattern_c0 : pattern_c0 + 3] = pattern
        sprite[:length, shaft_c] = color
        tip_r = 0
    sprite[tip_r, shaft_c - 1] = color
    sprite[tip_r, shaft_c + 1] = color
    return sprite


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _place_sprites(top_h: int, width: int, sprites: list[np.ndarray]) -> list[tuple[int, int]] | None:
    occupied = np.zeros((top_h, width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(sprites)
    order = sorted(range(len(sprites)), key=lambda i: sprites[i].shape[0] * sprites[i].shape[1], reverse=True)

    for idx in order:
        sprite = sprites[idx]
        mask = sprite != BG
        sh, sw = sprite.shape
        dilated = _dilate(occupied)
        candidates: list[tuple[int, int]] = []
        for r0 in range(top_h - sh + 1):
            for c0 in range(width - sw + 1):
                if (dilated[r0 : r0 + sh, c0 : c0 + sw] & mask).any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        positions[idx] = random.choice(candidates)
        r0, c0 = positions[idx]
        occupied[r0 : r0 + sh, c0 : c0 + sw] |= mask

    return [(int(r0), int(c0)) for r0, c0 in positions]


def _bbox_from_anchor(anchor: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = anchor
    return r - 1, c - 1, r + 1, c + 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _visible_overlap_difference(
    stamp_a: dict,
    anchor_a: tuple[int, int],
    stamp_b: dict,
    anchor_b: tuple[int, int],
) -> bool:
    pa = _rotate_pattern(stamp_a["pattern"], stamp_a["direction"])
    pb = _rotate_pattern(stamp_b["pattern"], stamp_b["direction"])
    ar0, ac0 = anchor_a[0] - 1, anchor_a[1] - 1
    br0, bc0 = anchor_b[0] - 1, anchor_b[1] - 1
    for ra in range(3):
        for ca in range(3):
            rr, cc = ar0 + ra, ac0 + ca
            if not (br0 <= rr <= br0 + 2 and bc0 <= cc <= bc0 + 2):
                continue
            rb, cb = rr - br0, cc - bc0
            if int(pa[ra, ca]) != int(pb[rb, cb]):
                return True
    return False


def _sample_anchors(bottom_h: int, width: int, stamps: list[dict]) -> dict[int, tuple[int, int]] | None:
    positions = [(r, c) for r in range(1, bottom_h - 1) for c in range(1, width - 1)]
    if len(positions) < len(stamps):
        return None

    force_overlap = len(stamps) >= 2 and random.random() < 0.5
    anchors: dict[int, tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()
    overlap_pair: tuple[int, int] | None = None

    if force_overlap:
        i, j = random.sample(range(len(stamps)), 2)
        random.shuffle(positions)
        placed = False
        for p0 in positions:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr == 0 and dc == 0:
                        continue
                    p1 = (p0[0] + dr, p0[1] + dc)
                    if p1 not in positions:
                        continue
                    if not _visible_overlap_difference(stamps[i], p0, stamps[j], p1):
                        continue
                    anchors[stamps[i]["arrow_color"]] = p0
                    anchors[stamps[j]["arrow_color"]] = p1
                    used.update([p0, p1])
                    overlap_pair = (stamps[i]["arrow_color"], stamps[j]["arrow_color"])
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    remaining = stamps[:]
    random.shuffle(remaining)
    for stamp in remaining:
        color = stamp["arrow_color"]
        if color in anchors:
            continue
        candidates = positions[:]
        random.shuffle(candidates)
        chosen = None
        for pos in candidates:
            if pos in used:
                continue
            box = _bbox_from_anchor(pos)
            ok = True
            for other_color, other_pos in anchors.items():
                other_box = _bbox_from_anchor(other_pos)
                allow_overlap = (
                    overlap_pair is not None
                    and {color, other_color} == set(overlap_pair)
                )
                if not allow_overlap and _boxes_overlap(box, other_box):
                    ok = False
                    break
            if ok:
                chosen = pos
                break
        if chosen is None:
            return None
        anchors[color] = chosen
        used.add(chosen)

    if overlap_pair is None:
        return anchors

    colors = list(anchors)
    overlaps = 0
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            if _boxes_overlap(_bbox_from_anchor(anchors[colors[i]]), _bbox_from_anchor(anchors[colors[j]])):
                overlaps += 1
    if overlaps != 1:
        return None
    return anchors


def _assemble_input(top_h: int, bottom_h: int, width: int, stamps: list[dict], anchors: dict[int, tuple[int, int]]) -> np.ndarray:
    grid = np.full((top_h + 1 + bottom_h, width), BG, dtype=int)
    grid[top_h, :] = SEP

    sprites = [_render_stamp(stamp) for stamp in stamps]
    positions = _place_sprites(top_h, width, sprites)
    if positions is None:
        raise ValueError("failed to place upper stamps")
    for sprite, (r0, c0) in zip(sprites, positions):
        mask = sprite != BG
        patch = grid[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]]
        patch[mask] = sprite[mask]

    for color, (r, c) in anchors.items():
        grid[top_h + 1 + r, c] = color
    return grid


def _make_stamp_specs() -> list[dict]:
    count = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 2])[0]
    arrow_colors = random.sample(ARROW_COLORS, count)
    pattern_colors = [c for c in ARROW_COLORS if c not in arrow_colors]
    lengths = random.sample(range(2, 7), count)

    specs = []
    for arrow_color, length in zip(arrow_colors, lengths):
        specs.append(
            {
                "arrow_color": arrow_color,
                "length": length,
                "direction": random.choice(["up", "right", "down", "left"]),
                "attach": random.randint(0, 2),
                "pattern": _random_pattern(pattern_colors),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    stamps = _make_stamp_specs()
    sprites = [_render_stamp(stamp) for stamp in stamps]
    min_top_h = max(sprite.shape[0] for sprite in sprites)
    max_top_h = min(14, 29 - 7)
    if min_top_h > max_top_h:
        return None

    top_h = random.randint(min_top_h, max_top_h)
    bottom_h = random.randint(7, min(12, 29 - top_h))
    width = random.randint(max(16, max(sprite.shape[1] for sprite in sprites)), 24)

    anchors = _sample_anchors(bottom_h, width, stamps)
    if anchors is None:
        return None

    try:
        grid = _assemble_input(top_h, bottom_h, width, stamps, anchors)
    except ValueError:
        return None

    expected = _apply_stamps((bottom_h, width), stamps, anchors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-arrow puzzle")
