import random
from collections import Counter

import numpy as np


SEPARATOR_COLOR = 6
FRAME_WIDTH = 5
FRAME_COUNT = 5
GEN_RETRIES = 400
FRAME_HEIGHT_RANGE = (6, 10)
SPRITE_SIZE_CHOICES = (3, 4, 5)
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rotate_clockwise(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, (-int(quarter_turns)) % 4).copy()


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            sizes.append(size)
    return sizes


def _find_separator_and_slices(grid: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for color in sorted(set(int(v) for v in np.unique(grid))):
        sep_cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if not sep_cols:
            continue
        sep_mask = np.zeros_like(grid, dtype=bool)
        sep_mask[:, sep_cols] = True
        if np.any((grid == color) & ~sep_mask):
            continue
        sep_set = set(sep_cols)
        slices: list[tuple[int, int]] = []
        c = 0
        while c < w:
            while c < w and c in sep_set:
                c += 1
            start = c
            while c < w and c not in sep_set:
                c += 1
            if start < c:
                slices.append((start, c))
        widths = {c1 - c0 for c0, c1 in slices}
        if len(slices) >= 3 and len(widths) == 1:
            candidates.append((color, slices))

    assert candidates, "expected at least one valid separator partition"
    assert len(candidates) == 1, "separator partition must be unique"
    sep_color, slices = candidates[0]
    return int(sep_color), slices


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_color, slices = _find_separator_and_slices(g)
    frames = [g[:, c0:c1].copy() for c0, c1 in slices]
    assert len(frames) >= 4
    frame_w = frames[0].shape[1]
    assert all(frame.shape[1] == frame_w for frame in frames)

    non_sep = np.concatenate([frame.ravel() for frame in frames])
    bg = int(Counter(map(int, non_sep)).most_common(1)[0][0])

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg, sep_color})
    assert colors, "expected foreground colors besides background and separator"
    singleton_colors = [color for color in colors if max(_component_sizes(g == color), default=0) == 1]
    assert len(singleton_colors) == 1, "dot color must be uniquely identifiable as singleton-only"
    dot_color = singleton_colors[0]

    roles = []
    for idx, frame in enumerate(frames):
        dot_coords = np.argwhere(frame == dot_color)
        dot_count = int(len(dot_coords))
        sprite_mask = (frame != bg) & (frame != dot_color)
        sprite_bbox = _bbox(sprite_mask)
        roles.append(
            {
                "index": idx,
                "frame": frame,
                "dot_count": dot_count,
                "dot_coords": [(int(r), int(c)) for r, c in dot_coords],
                "has_sprite": bool(sprite_bbox is not None),
                "sprite_bbox": sprite_bbox,
            }
        )

    answer_frames = [role for role in roles if not role["has_sprite"]]
    assert len(answer_frames) == 1, "expected exactly one frame with dots but no sprite"
    answer = answer_frames[0]
    assert 1 <= answer["dot_count"] <= 3, "answer frame must encode a non-zero quarter-turn count"

    observed = [role for role in roles if role["has_sprite"]]
    baseline_candidates = [role for role in observed if role["dot_count"] == 0]
    assert len(baseline_candidates) == 1, "expected exactly one zero-dot baseline frame"
    baseline = baseline_candidates[0]

    bboxes = {role["sprite_bbox"] for role in observed}
    assert len(bboxes) == 1, "sprite anchor bbox must be identical across visible frames"
    r0, c0, r1, c1 = baseline["sprite_bbox"]
    size = r1 - r0 + 1
    assert size == c1 - c0 + 1, "sprite anchor must be square"

    for role in roles:
        for rr, cc in role["dot_coords"]:
            assert not (r0 <= rr <= r1 and c0 <= cc <= c1), "dots must stay outside the sprite anchor"
        assert role["dot_count"] <= 3, "rotation code uses 0..3 sparse dots"

    base_sprite = baseline["frame"][r0 : r1 + 1, c0 : c1 + 1].copy()
    rotated_signatures = {_rotate_clockwise(base_sprite, k).tobytes() for k in range(4)}
    assert len(rotated_signatures) == 4, "base sprite must have four distinct rotations"

    for role in observed:
        expected = _rotate_clockwise(base_sprite, role["dot_count"])
        actual = role["frame"][r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(actual, expected), "visible frames must obey the count-to-rotation rule"

    return {
        "bg": bg,
        "sep_color": sep_color,
        "dot_color": dot_color,
        "frames": frames,
        "roles": roles,
        "answer": answer,
        "baseline": baseline,
        "anchor_bbox": (r0, c0, r1, c1),
        "base_sprite": base_sprite,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing frame by rotating the shared sprite according to that frame's sparse-dot count.

    Input structure:
    - Vertical separator columns split the grid into equal-width frames.
    - Each visible frame contains the same sprite anchored in the same square bbox plus 0-3 isolated dots,
      except one answer frame that contains only dots and no sprite.
    - Exactly one visible sprite frame has zero dots; that frame is the baseline orientation.

    Rule:
    - In every visible sprite frame, the dot count encodes the sprite rotation: 0, 1, 2, 3 dots mean
      0°, 90°, 180°, 270° clockwise relative to the zero-dot baseline.
    - Ignore frame order entirely; it is shuffled.
    - Output the answer frame alone, preserving its dots and placing the correctly rotated sprite back
      into the shared anchor bbox.

    Key insight:
    - The sparse dots are not decoration and they are not a sequence across columns; each frame carries
      its own rotation instruction, and the zero-dot frame bootstraps the codebook.
    """
    info = _analyze_grid(grid)
    answer_frame = info["answer"]["frame"].copy()
    r0, c0, r1, c1 = info["anchor_bbox"]
    sprite = _rotate_clockwise(info["base_sprite"], info["answer"]["dot_count"])
    answer_frame[r0 : r1 + 1, c0 : c1 + 1] = sprite
    return answer_frame


def _random_sprite_mask(size: int) -> np.ndarray | None:
    min_area = max(size + 1, int(np.ceil(size * size * 0.42)))
    max_area = size * size - 1
    for _ in range(120):
        prob = random.uniform(0.42, 0.68)
        mask = np.random.rand(size, size) < prob
        area = int(mask.sum())
        if not (min_area <= area <= max_area):
            continue
        if not bool(mask.any(axis=0).all()) or not bool(mask.any(axis=1).all()):
            continue
        if _component_sizes(mask) != [area]:
            continue
        if len({np.rot90(mask, k).tobytes() for k in range(4)}) != 4:
            continue
        return mask
    return None


def _sample_dots(frame_h: int, frame_w: int, bbox: tuple[int, int, int, int], count: int) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    r0, c0, r1, c1 = bbox
    candidates = [
        (r, c)
        for r in range(frame_h)
        for c in range(frame_w)
        if not (r0 <= r <= r1 and c0 <= c <= c1)
    ]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        if all(max(abs(cell[0] - rr), abs(cell[1] - cc)) > 1 for rr, cc in chosen):
            chosen.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _order_is_too_orderly(counts: list[int]) -> bool:
    return counts == sorted(counts) or counts == sorted(counts, reverse=True)


def _build_frame(
    frame_h: int,
    frame_w: int,
    bg: int,
    dot_color: int,
    dots: list[tuple[int, int]],
    sprite: np.ndarray | None,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    frame = np.full((frame_h, frame_w), bg, dtype=int)
    if sprite is not None:
        r0, c0, r1, c1 = bbox
        frame[r0 : r1 + 1, c0 : c1 + 1] = sprite
    for r, c in dots:
        frame[r, c] = dot_color
    return frame


def _validate_example(grid: np.ndarray, expected: np.ndarray, answer_count: int) -> bool:
    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return False
    observed_counts = sorted(role["dot_count"] for role in info["roles"] if role["has_sprite"])
    if observed_counts != [0, 1, 2, 3]:
        return False
    if info["answer"]["dot_count"] != answer_count:
        return False
    if not np.array_equal(main(grid), expected):
        return False
    counts_in_order = [role["dot_count"] for role in info["roles"]]
    if _order_is_too_orderly(counts_in_order):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    sprite_color, dot_color = random.sample(PALETTE, 2)
    frame_h = random.randint(*FRAME_HEIGHT_RANGE)
    frame_w = FRAME_WIDTH
    sprite_size = random.choice([size for size in SPRITE_SIZE_CHOICES if size <= frame_h and size <= frame_w])
    bbox_r0 = random.randint(0, frame_h - sprite_size)
    bbox_c0 = random.randint(0, frame_w - sprite_size)
    bbox = (bbox_r0, bbox_c0, bbox_r0 + sprite_size - 1, bbox_c0 + sprite_size - 1)

    base_mask = _random_sprite_mask(sprite_size)
    if base_mask is None:
        return None
    base_sprite = np.full((sprite_size, sprite_size), bg, dtype=int)
    base_sprite[base_mask] = sprite_color

    observed_dots: dict[int, list[tuple[int, int]]] = {}
    for count in (0, 1, 2, 3):
        dots = _sample_dots(frame_h, frame_w, bbox, count)
        if dots is None:
            return None
        observed_dots[count] = dots

    answer_count = random.randint(1, 3)
    answer_dots = _sample_dots(frame_h, frame_w, bbox, answer_count)
    if answer_dots is None or sorted(answer_dots) == sorted(observed_dots[answer_count]):
        return None

    frames_meta = []
    for count in (0, 1, 2, 3):
        sprite = _rotate_clockwise(base_sprite, count)
        frames_meta.append(
            {
                "count": count,
                "has_sprite": True,
                "dots": observed_dots[count],
                "frame": _build_frame(frame_h, frame_w, bg, dot_color, observed_dots[count], sprite, bbox),
            }
        )

    answer_sprite = _rotate_clockwise(base_sprite, answer_count)
    expected = _build_frame(frame_h, frame_w, bg, dot_color, answer_dots, answer_sprite, bbox)
    frames_meta.append(
        {
            "count": answer_count,
            "has_sprite": False,
            "dots": answer_dots,
            "frame": _build_frame(frame_h, frame_w, bg, dot_color, answer_dots, None, bbox),
        }
    )

    random.shuffle(frames_meta)
    if _order_is_too_orderly([meta["count"] for meta in frames_meta]):
        return None

    full_w = FRAME_COUNT * frame_w + (FRAME_COUNT - 1)
    grid = np.full((frame_h, full_w), bg, dtype=int)
    col = 0
    for idx, meta in enumerate(frames_meta):
        grid[:, col : col + frame_w] = meta["frame"]
        col += frame_w
        if idx != FRAME_COUNT - 1:
            grid[:, col] = SEPARATOR_COLOR
            col += 1

    if not _validate_example(grid, expected, answer_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid frame-sequence puzzle after {GEN_RETRIES} attempts")
