from __future__ import annotations

import random
from collections import Counter

import numpy as np


FRAME_SIZE = 5
INNER_SIZE = 3
BACKGROUND = 0
SPRITE_COLORS = (1, 3)
BORDER_COLORS = (2, 4, 5, 6, 7, 8, 9)
GRID_MIN = 18
GRID_MAX = 24
FRAME_GAP = 2
SPRITE_MIN_NONZERO = 3
SPRITE_MAX_NONZERO = 6
MAX_GENERATION_ATTEMPTS = 400

TRANSFORM_NAMES = (
    "rotate_cw",
    "rotate_ccw",
    "flip_horizontal",
    "flip_vertical",
    "rotate_180",
)


def _border_mask() -> np.ndarray:
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


BORDER_MASK = _border_mask()


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid).reshape(-1)))
    return int(counts.most_common(1)[0][0])


def _anchor_key(sprite: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(np.asarray(sprite) != BACKGROUND)
    assert coords.size > 0, "sprite must contain at least one non-background pixel"
    rows = coords[:, 0]
    top = coords[rows == rows.min()]
    leftmost = int(top[:, 1].min())
    return int(rows.min()), leftmost


def _apply_transform(sprite: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(sprite, dtype=int)
    if name == "rotate_cw":
        return np.rot90(arr, -1).copy()
    if name == "rotate_ccw":
        return np.rot90(arr, 1).copy()
    if name == "flip_horizontal":
        return np.fliplr(arr).copy()
    if name == "flip_vertical":
        return np.flipud(arr).copy()
    if name == "rotate_180":
        return np.rot90(arr, 2).copy()
    raise ValueError(f"unknown transform {name}")


def _matching_transforms(src: np.ndarray, dst: np.ndarray) -> list[str]:
    return [name for name in TRANSFORM_NAMES if np.array_equal(_apply_transform(src, name), dst)]


def _sprite_signature(sprite: np.ndarray) -> bytes:
    return np.asarray(sprite, dtype=int).tobytes()


def _make_frame(border_color: int, sprite: np.ndarray) -> np.ndarray:
    patch = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=int)
    patch[BORDER_MASK] = int(border_color)
    patch[1:-1, 1:-1] = np.asarray(sprite, dtype=int)
    return patch


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    height, width = g.shape
    frames: list[dict] = []

    for r0 in range(height - FRAME_SIZE + 1):
        for c0 in range(width - FRAME_SIZE + 1):
            patch = g[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE]
            border_color = int(patch[0, 0])
            if border_color == bg:
                continue
            if not np.all(patch[BORDER_MASK] == border_color):
                continue
            inner = patch[1:-1, 1:-1].copy()
            assert np.any(inner != bg), "every frame must contain a non-empty sprite"
            frames.append(
                {
                    "bbox": (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1),
                    "border_color": border_color,
                    "inner": inner,
                    "anchor": _anchor_key(inner),
                }
            )

    assert len(frames) == 3, "expected exactly three 5x5 frames"
    return sorted(frames, key=lambda frame: frame["bbox"])


def _analyze(grid: np.ndarray) -> dict:
    frames = _extract_frames(grid)
    groups: dict[int, list[dict]] = {}
    for frame in frames:
        groups.setdefault(int(frame["border_color"]), []).append(frame)

    pair_colors = [color for color, items in groups.items() if len(items) == 2]
    single_colors = [color for color, items in groups.items() if len(items) == 1]
    assert len(pair_colors) == 1 and len(single_colors) == 1, "expected one training color pair and one test color"

    train_color = int(pair_colors[0])
    test_color = int(single_colors[0])
    a, b = groups[train_color]
    assert a["anchor"] != b["anchor"], "training direction must be uniquely determined by local sprite anchors"
    source, target = (a, b) if a["anchor"] < b["anchor"] else (b, a)

    matches = _matching_transforms(source["inner"], target["inner"])
    assert len(matches) == 1, "training pair must encode exactly one allowed transform"

    return {
        "frames": frames,
        "train_color": train_color,
        "test_color": test_color,
        "source": source,
        "target": target,
        "test": groups[test_color][0],
        "transform_name": matches[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a border-color-coded geometric transform from a paired set of framed sprites and apply it to the test sprite.

    Input structure:
    - A black background grid contains exactly three disjoint 5x5 hollow square frames.
    - Two frames share the same border color and contain 3x3 colored sprites; these form the training pair.
    - The remaining frame has a different border color and contains a third 3x3 colored sprite for testing.

    Rule:
    - Group frames by border color; the color appearing twice identifies the training pair and the singleton color identifies the test frame.
    - Inside the training pair, compare the local 3x3 sprites and order them by the topmost non-black pixel, breaking ties by the leftmost such pixel.
    - The unique transform mapping the earlier sprite to the later sprite must be one of: clockwise 90-degree rotation, counterclockwise 90-degree rotation, horizontal mirror, vertical mirror, or 180-degree rotation.
    - Apply that same transform to the test frame's 3x3 sprite and return only the transformed 3x3 crop.

    Key insight:
    - The repeated border color is not decorative; it marks the example pair whose internal local-coordinate relation defines the transform that must be transferred to the differently colored test sprite.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    return _apply_transform(info["test"]["inner"], info["transform_name"])


def _build_base_sprites() -> tuple[np.ndarray, ...]:
    values = (BACKGROUND, *SPRITE_COLORS)
    valid: list[np.ndarray] = []

    for code in range(len(values) ** (INNER_SIZE * INNER_SIZE)):
        sprite = np.zeros((INNER_SIZE, INNER_SIZE), dtype=int)
        x = code
        seen_colors: set[int] = set()
        nonzero = 0
        for idx in range(INNER_SIZE * INNER_SIZE):
            value = int(values[x % len(values)])
            x //= len(values)
            sprite[idx // INNER_SIZE, idx % INNER_SIZE] = value
            if value != BACKGROUND:
                nonzero += 1
                seen_colors.add(value)

        if seen_colors != set(SPRITE_COLORS):
            continue
        if not (SPRITE_MIN_NONZERO <= nonzero <= SPRITE_MAX_NONZERO):
            continue

        transformed = [_apply_transform(sprite, name) for name in TRANSFORM_NAMES]
        if any(np.array_equal(other, sprite) for other in transformed):
            continue
        if len({_sprite_signature(other) for other in transformed}) != len(TRANSFORM_NAMES):
            continue
        valid.append(sprite)

    assert valid, "expected at least one asymmetric sprite"
    return tuple(valid)


BASE_SPRITES = _build_base_sprites()
TRAINING_SPRITES = {
    name: tuple(
        sprite
        for sprite in BASE_SPRITES
        if _anchor_key(sprite) < _anchor_key(_apply_transform(sprite, name))
    )
    for name in TRANSFORM_NAMES
}
assert all(TRAINING_SPRITES.values()), "every transform needs valid training sprites"


def _boxes_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    return (
        ar0 + FRAME_SIZE + FRAME_GAP <= br0
        or br0 + FRAME_SIZE + FRAME_GAP <= ar0
        or ac0 + FRAME_SIZE + FRAME_GAP <= bc0
        or bc0 + FRAME_SIZE + FRAME_GAP <= ac0
    )


def _sample_positions(height: int, width: int) -> list[tuple[int, int]]:
    candidates = [
        (r0, c0)
        for r0 in range(height - FRAME_SIZE + 1)
        for c0 in range(width - FRAME_SIZE + 1)
    ]
    for _ in range(200):
        order = random.sample(candidates, len(candidates))
        chosen: list[tuple[int, int]] = []
        for pos in order:
            if all(_boxes_separated(pos, other) for other in chosen):
                chosen.append(pos)
                if len(chosen) == 3:
                    return chosen
    raise RuntimeError("failed to place three separated frames")


def _pick_test_sprite(transform_name: str, train_source: np.ndarray, train_target: np.ndarray) -> np.ndarray:
    blocked = {_sprite_signature(train_source), _sprite_signature(train_target)}
    candidates = []
    for sprite in BASE_SPRITES:
        transformed = _apply_transform(sprite, transform_name)
        out_sig = _sprite_signature(transformed)
        if _sprite_signature(sprite) in blocked:
            continue
        if out_sig in blocked:
            continue
        candidates.append(sprite)
    assert candidates, "expected at least one test sprite candidate"
    return random.choice(candidates)


def _render_instance(
    height: int,
    width: int,
    train_color: int,
    test_color: int,
    source_sprite: np.ndarray,
    target_sprite: np.ndarray,
    test_sprite: np.ndarray,
    source_globally_later: bool,
) -> tuple[np.ndarray, dict]:
    positions = _sample_positions(height, width)
    test_pos = random.choice(positions)
    train_positions = sorted(pos for pos in positions if pos != test_pos)
    if source_globally_later:
        source_pos, target_pos = train_positions[1], train_positions[0]
    else:
        source_pos, target_pos = train_positions[0], train_positions[1]

    grid = np.zeros((height, width), dtype=int)
    for pos, border_color, sprite in (
        (source_pos, train_color, source_sprite),
        (target_pos, train_color, target_sprite),
        (test_pos, test_color, test_sprite),
    ):
        r0, c0 = pos
        grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = _make_frame(border_color, sprite)

    return grid, {
        "source_pos": source_pos,
        "target_pos": target_pos,
        "test_pos": test_pos,
        "source_globally_later": source_globally_later,
    }


def _validate_instance(grid: np.ndarray, expected: dict) -> bool:
    try:
        analysis = _analyze(grid)
    except AssertionError:
        return False

    source = analysis["source"]["inner"]
    target = analysis["target"]["inner"]
    test = analysis["test"]["inner"]
    output = _apply_transform(test, analysis["transform_name"])

    if analysis["transform_name"] != expected["transform_name"]:
        return False
    if not np.array_equal(source, expected["source_sprite"]):
        return False
    if not np.array_equal(target, expected["target_sprite"]):
        return False
    if not np.array_equal(test, expected["test_sprite"]):
        return False
    if not np.array_equal(output, expected["expected_output"]):
        return False
    if not np.array_equal(main(grid), expected["expected_output"]):
        return False
    if np.array_equal(test, expected["expected_output"]):
        return False
    if np.array_equal(expected["expected_output"], expected["source_sprite"]):
        return False
    if np.array_equal(expected["expected_output"], expected["target_sprite"]):
        return False

    other_hits = sum(
        np.array_equal(_apply_transform(test, name), expected["expected_output"])
        for name in TRANSFORM_NAMES
    )
    if other_hits != 1:
        return False

    train_bboxes = [frame["bbox"][:2] for frame in analysis["frames"] if frame["border_color"] == analysis["train_color"]]
    train_bboxes.sort()
    if expected["source_globally_later"]:
        if analysis["source"]["bbox"][:2] != train_bboxes[1]:
            return False
    else:
        if analysis["source"]["bbox"][:2] != train_bboxes[0]:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        transform_name = random.choice(TRANSFORM_NAMES)
        source_sprite = random.choice(TRAINING_SPRITES[transform_name])
        target_sprite = _apply_transform(source_sprite, transform_name)
        test_sprite = _pick_test_sprite(transform_name, source_sprite, target_sprite)
        expected_output = _apply_transform(test_sprite, transform_name)

        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)
        train_color, test_color = random.sample(BORDER_COLORS, 2)
        source_globally_later = random.choice([False, True])

        grid, placement = _render_instance(
            height,
            width,
            train_color,
            test_color,
            source_sprite,
            target_sprite,
            test_sprite,
            source_globally_later,
        )
        expected = {
            "transform_name": transform_name,
            "source_sprite": source_sprite,
            "target_sprite": target_sprite,
            "test_sprite": test_sprite,
            "expected_output": expected_output,
            "source_globally_later": placement["source_globally_later"],
        }
        if _validate_instance(grid, expected):
            return grid

    raise RuntimeError("failed to generate a valid input")
