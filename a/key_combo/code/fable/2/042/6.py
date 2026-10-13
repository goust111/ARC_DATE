from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_coords(coords)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 3 or ww < 3:
        return False
    if int(coords.shape[0]) != 2 * (hh + ww) - 4:
        return False
    local = np.zeros((hh, ww), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((hh, ww), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(local, expected))


def _parse_palette(row: np.ndarray, bg: int) -> dict[int, int]:
    row = np.asarray(row, dtype=int)
    mapping: dict[int, int] = {}
    c = 0
    while c < row.size:
        if int(row[c]) == int(bg):
            c += 1
            continue
        start = c
        while c < row.size and int(row[c]) != int(bg):
            c += 1
        run = row[start:c]
        assert run.size == 2, "each palette entry must be one adjacent color pair"
        a, b = map(int, run)
        assert a != b, "palette pair colors must differ"
        assert a not in mapping and b not in mapping, "palette colors must appear in exactly one pair"
        mapping[a] = b
        mapping[b] = a
    assert len(mapping) >= 6, "expected at least three palette pairs"
    return mapping


def _find_box_frames(grid: np.ndarray, bg: int) -> dict[int, dict]:
    boxes: dict[int, dict] = {}
    for color in sorted(set(map(int, grid.reshape(-1)))):
        if color in (int(bg), GREY):
            continue
        for comp in _find_components_of_color(grid, color):
            if not _is_rect_frame(comp["coords"]):
                continue
            r0, c0, r1, c1 = comp["bbox"]
            if r0 <= 0:
                continue
            interior = _crop_bbox(grid, (r0 + 1, c0 + 1, r1 - 1, c1 - 1))
            assert color not in boxes, "box border colors must be unique"
            boxes[color] = {
                "bbox": comp["bbox"],
                "interior": interior,
            }
    assert len(boxes) >= 4, "expected several colored boxes"
    return boxes


def _solve(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    mapping = _parse_palette(g[0], bg)

    grey_frames = [
        comp
        for comp in _find_components_of_color(g, GREY)
        if _is_rect_frame(comp["coords"]) and comp["bbox"][0] > 0
    ]
    assert len(grey_frames) == 1, "expected exactly one grey rectangular frame"
    frame = grey_frames[0]
    r0, c0, r1, c1 = frame["bbox"]
    assert r1 - r0 >= 3 and c1 - c0 >= 3, "grey frame must have a nontrivial interior"

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.size > 0, "grey frame interior must be nonempty"
    counts = Counter(int(v) for v in inner.reshape(-1) if int(v) not in (int(bg), GREY))
    assert counts, "grey frame interior must contain colored noise"
    ranked = counts.most_common()
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "frame noise must have a unique majority color"
    majority_color = int(ranked[0][0])
    assert majority_color in mapping, "frame majority color must be present in the palette"
    partner_color = int(mapping[majority_color])

    boxes = _find_box_frames(g, bg)
    assert majority_color in boxes, "expected a decoy box using the majority color"
    assert partner_color in boxes, "expected a target box using the paired color"

    target = boxes[partner_color]["interior"]
    return {
        "target": target,
        "bg": int(bg),
        "mapping": mapping,
        "majority": majority_color,
        "partner": partner_color,
        "boxes": boxes,
        "frame_bbox": frame["bbox"],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the unique majority noise color inside the grey frame, map it through the top-row palette pair, and crop the inside of the matched small box.

    Input structure:
    - The top row is a palette made of separated adjacent two-color pairs on the background; each pair defines a bidirectional color mapping.
    - Exactly one grey rectangular frame appears below the palette, and its interior contains scattered colored noise pixels with one unique majority color.
    - Several other small rectangular boxes with distinct border colors lie elsewhere; one box border is the paired color from the palette, while another decoy box border is the majority color itself.

    Rule:
    - Read the unique majority non-background noise color inside the grey frame.
    - In the top-row palette, find the color pair containing that majority color and take the opposite color in the pair.
    - Find the small box whose border uses that paired color and return only its interior crop, excluding the border.

    Key insight:
    - The grey frame supplies a first key by counting, the palette translates that key, and only the translated color indexes the correct box; using the majority color directly lands on the decoy.
    """
    return _solve(grid)["target"].copy()


def _random_free_location(
    occupied: np.ndarray,
    hh: int,
    ww: int,
    *,
    min_row: int = 2,
    border: int = 1,
    padding: int = 1,
) -> tuple[int, int]:
    h, w = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(min_row, h - border - hh + 1):
        for c0 in range(border, w - border - ww + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + hh + padding)
            cc1 = min(w, c0 + ww + padding)
            if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no free location")
    return random.choice(candidates)


def _make_pattern(
    hh: int,
    ww: int,
    *,
    forbidden: set[int],
    existing: set[tuple[tuple[int, int], bytes]],
    heavy_color: int | None = None,
) -> np.ndarray | None:
    allowed = [c for c in PALETTE_COLORS if c not in forbidden]
    if not allowed:
        return None
    area = hh * ww
    for _ in range(80):
        if heavy_color in allowed and random.random() < 0.75:
            base = int(heavy_color)
        else:
            base = int(random.choice(allowed))
        patch = np.full((hh, ww), base, dtype=int)
        others = [c for c in allowed if c != base]
        if others and area >= 2:
            accent1 = int(random.choice(others))
            k1 = random.randint(1, max(1, area // 2))
            idx1 = random.sample(range(area), k1)
            for idx in idx1:
                patch[idx // ww, idx % ww] = accent1
            if len(others) >= 2 and area >= 4 and random.random() < 0.45:
                accent2 = int(random.choice([c for c in others if c != accent1]))
                free = [idx for idx in range(area) if patch[idx // ww, idx % ww] != accent2]
                k2 = random.randint(1, max(1, area // 3))
                for idx in random.sample(free, min(k2, len(free))):
                    patch[idx // ww, idx % ww] = accent2
        sig = (patch.shape, patch.tobytes())
        if sig in existing:
            continue
        existing.add(sig)
        return patch
    return None


def _make_box_patch(border_color: int, interior: np.ndarray) -> np.ndarray:
    hh, ww = interior.shape
    patch = np.full((hh + 2, ww + 2), int(border_color), dtype=int)
    patch[1:-1, 1:-1] = interior
    return patch


def _make_noise_patch(
    inner_h: int,
    inner_w: int,
    majority_color: int,
) -> np.ndarray | None:
    area = inner_h * inner_w
    if area < 12:
        return None
    other_choices = [c for c in PALETTE_COLORS if c != int(majority_color)]
    other_k = random.randint(2, min(4, len(other_choices)))
    other_colors = random.sample(other_choices, other_k)
    majority_min = 4
    majority_max = min(area - other_k - 1, max(5, area // 2))
    if majority_max < majority_min:
        return None
    majority_count = random.randint(majority_min, majority_max)
    total_noise = random.randint(majority_count + other_k, min(area - 1, majority_count + other_k * 3))
    other_counts = [1] * other_k
    remaining = total_noise - majority_count - other_k
    while remaining > 0:
        idx = random.randrange(other_k)
        if other_counts[idx] < min(3, majority_count - 1):
            other_counts[idx] += 1
            remaining -= 1
    if max(other_counts) >= majority_count:
        return None
    patch = np.full((inner_h, inner_w), BG, dtype=int)
    positions = random.sample(range(area), total_noise)
    colors = [majority_color] * majority_count
    for color, count in zip(other_colors, other_counts):
        colors.extend([int(color)] * int(count))
    random.shuffle(colors)
    for pos, color in zip(positions, colors):
        patch[pos // inner_w, pos % inner_w] = int(color)
    return patch


def _draw_palette(grid: np.ndarray, pairs: list[tuple[int, int]]) -> None:
    seps = [random.randint(1, 2) for _ in range(len(pairs) - 1)]
    total_len = 2 * len(pairs) + sum(seps)
    start = random.randint(1, grid.shape[1] - total_len - 1)
    c = start
    for idx, (left, right) in enumerate(pairs):
        grid[0, c] = int(left)
        grid[0, c + 1] = int(right)
        c += 2
        if idx < len(seps):
            c += seps[idx]


def _validate_semantics(
    grid: np.ndarray,
    target_pattern: np.ndarray,
    majority_color: int,
    partner_color: int,
) -> bool:
    if _get_bg(grid) != BG:
        return False
    try:
        info = _solve(grid)
    except AssertionError:
        return False
    if info["majority"] != int(majority_color) or info["partner"] != int(partner_color):
        return False
    if not np.array_equal(info["target"], target_pattern):
        return False
    if np.any(target_pattern == int(majority_color)):
        return False

    box_patterns = {color: data["interior"] for color, data in info["boxes"].items()}
    pattern_sigs = {(arr.shape, arr.tobytes()) for arr in box_patterns.values()}
    if len(pattern_sigs) != len(box_patterns):
        return False
    if np.array_equal(box_patterns[int(majority_color)], target_pattern):
        return False

    global_counts = Counter(
        int(v) for v in grid.reshape(-1) if int(v) not in (BG, GREY)
    )
    ranked = global_counts.most_common()
    if ranked and ranked[0][0] == int(majority_color):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.choice((3, 4))
    palette_colors = random.sample(PALETTE_COLORS, 2 * pair_count)
    random.shuffle(palette_colors)
    pairs = [
        (int(palette_colors[2 * i]), int(palette_colors[2 * i + 1]))
        for i in range(pair_count)
    ]
    key_pair_idx = random.randrange(pair_count)
    key_left, key_right = pairs[key_pair_idx]
    if random.random() < 0.5:
        majority_color, partner_color = key_left, key_right
    else:
        majority_color, partner_color = key_right, key_left

    remaining = [c for pair in pairs for c in pair if c not in (majority_color, partner_color)]
    box_count = random.randint(4, min(5, len(remaining) + 2))
    extra_colors = random.sample(remaining, box_count - 2)
    border_colors = [majority_color, partner_color, *extra_colors]
    random.shuffle(border_colors)
    heavy_color = random.choice(extra_colors)

    target_inner_shape = (random.randint(2, 5), random.randint(2, 5))
    same_decoy_size = random.random() < 0.65
    existing_patterns: set[tuple[tuple[int, int], bytes]] = set()
    boxes_spec: list[dict] = []
    target_pattern: np.ndarray | None = None

    for border_color in border_colors:
        if border_color == partner_color:
            inner_h, inner_w = target_inner_shape
        elif border_color == majority_color and same_decoy_size:
            inner_h, inner_w = target_inner_shape
        else:
            inner_h, inner_w = random.randint(2, 5), random.randint(2, 5)

        pattern = _make_pattern(
            inner_h,
            inner_w,
            forbidden={BG, GREY, border_color, majority_color},
            existing=existing_patterns,
            heavy_color=heavy_color if border_color != heavy_color else None,
        )
        if pattern is None:
            return None
        if border_color == partner_color:
            target_pattern = pattern.copy()
        boxes_spec.append(
            {
                "border_color": int(border_color),
                "interior": pattern,
                "patch": _make_box_patch(int(border_color), pattern),
            }
        )

    assert target_pattern is not None

    frame_h = random.randint(6, 9)
    frame_w = random.randint(6, 9)
    noise_patch = _make_noise_patch(frame_h - 2, frame_w - 2, majority_color)
    if noise_patch is None:
        return None

    h = random.randint(22, 29)
    w = random.randint(max(22, 3 * pair_count + 7), 29)
    grid = np.full((h, w), BG, dtype=int)
    _draw_palette(grid, pairs)

    occupied = np.zeros((h, w), dtype=bool)
    objects = [{"kind": "frame", "hh": frame_h, "ww": frame_w}]
    for spec in boxes_spec:
        ph, pw = spec["patch"].shape
        objects.append({"kind": "box", "hh": ph, "ww": pw, "spec": spec})
    random.shuffle(objects)
    objects.sort(key=lambda obj: obj["hh"] * obj["ww"], reverse=True)

    placements: list[dict] = []
    for obj in objects:
        try:
            r0, c0 = _random_free_location(occupied, obj["hh"], obj["ww"], min_row=2, border=1, padding=1)
        except ValueError:
            return None
        occupied[r0 : r0 + obj["hh"], c0 : c0 + obj["ww"]] = True
        placements.append({**obj, "r0": r0, "c0": c0})

    for obj in placements:
        r0, c0 = obj["r0"], obj["c0"]
        if obj["kind"] == "frame":
            grid[r0, c0 : c0 + frame_w] = GREY
            grid[r0 + frame_h - 1, c0 : c0 + frame_w] = GREY
            grid[r0 : r0 + frame_h, c0] = GREY
            grid[r0 : r0 + frame_h, c0 + frame_w - 1] = GREY
            grid[r0 + 1 : r0 + frame_h - 1, c0 + 1 : c0 + frame_w - 1] = noise_patch
        else:
            patch = obj["spec"]["patch"]
            ph, pw = patch.shape
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch

    if not _validate_semantics(grid, target_pattern, majority_color, partner_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid indirect-addressing ARC input")
