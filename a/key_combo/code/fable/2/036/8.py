from __future__ import annotations

import math
import random

import numpy as np


ALL_COLORS = tuple(range(10))
OUTPUT_COLORS = tuple(range(1, 10))
SCALES = (4, 6)
TIE_PROBABILITY = 0.55
MAX_GENERATION_ATTEMPTS = 500


def _divisors_gt1(n: int) -> list[int]:
    return [d for d in range(2, int(n) + 1) if int(n) % d == 0]


def _proper_divisors(scale: int) -> list[int]:
    return [d for d in _divisors_gt1(scale) if d < scale]


def _candidate_scales(shape: tuple[int, int]) -> list[int]:
    height, width = map(int, shape)
    return [d for d in range(2, min(height, width) + 1) if height % d == 0 and width % d == 0]


def _scale_up(grid: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(np.asarray(grid, dtype=int), np.ones((scale, scale), dtype=int))


def _block_status(block: np.ndarray) -> tuple[str, int | None, tuple[int, int] | None]:
    colors, counts = np.unique(np.asarray(block, dtype=int), return_counts=True)
    order = np.argsort(-counts)
    colors = colors[order]
    counts = counts[order]
    area = int(block.size)
    top_count = int(counts[0])

    if top_count * 2 > area:
        return "majority", int(colors[0]), None
    if top_count * 2 == area and len(counts) >= 2 and int(counts[1]) == top_count:
        pair = tuple(sorted((int(colors[0]), int(colors[1]))))
        return "tie", None, pair
    return "invalid", None, None


def _analyze_scale(grid: np.ndarray, scale: int) -> dict:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    if height % scale != 0 or width % scale != 0:
        return {"scale": scale, "valid": False, "colors": None, "ties": []}

    small_h = height // scale
    small_w = width // scale
    restored = np.full((small_h, small_w), -1, dtype=int)
    ties: list[tuple[int, int, tuple[int, int]]] = []

    for rr in range(small_h):
        for cc in range(small_w):
            block = g[rr * scale : (rr + 1) * scale, cc * scale : (cc + 1) * scale]
            status, color, pair = _block_status(block)
            if status == "majority":
                restored[rr, cc] = int(color)
            elif status == "tie":
                ties.append((rr, cc, pair))
            else:
                return {"scale": scale, "valid": False, "colors": None, "ties": []}

    if len(ties) > 1:
        return {"scale": scale, "valid": False, "colors": None, "ties": []}

    if ties:
        rr, cc, pair = ties[0]
        mirror_c = small_w - 1 - cc
        if mirror_c == cc:
            return {"scale": scale, "valid": False, "colors": None, "ties": []}
        mirror_color = int(restored[rr, mirror_c])
        if mirror_color < 0 or mirror_color not in pair:
            return {"scale": scale, "valid": False, "colors": None, "ties": []}
        restored[rr, cc] = mirror_color

    if not np.array_equal(restored, np.fliplr(restored)):
        return {"scale": scale, "valid": False, "colors": None, "ties": []}

    return {
        "scale": scale,
        "valid": True,
        "colors": restored,
        "ties": [(rr, cc) for rr, cc, _ in ties],
    }


def _noise_colors(exclude: int, count: int) -> list[int]:
    choices = [color for color in ALL_COLORS if color != exclude]
    return random.sample(choices, count)


def _make_majority_block(scale: int, major_color: int, *, max_noise: int | None = None) -> np.ndarray:
    area = scale * scale
    max_noise = min(area // 2 - 1, max_noise if max_noise is not None else max(2, scale + 1))
    noise_count = random.randint(1, max_noise)
    noise_palette = _noise_colors(major_color, random.randint(1, min(3, len(ALL_COLORS) - 1)))
    block = np.full((scale, scale), int(major_color), dtype=int)
    for idx in random.sample(range(area), noise_count):
        r, c = divmod(idx, scale)
        block[r, c] = random.choice(noise_palette)
    status, color, _ = _block_status(block)
    assert status == "majority" and color == major_color
    return block


def _make_tie_block(scale: int, major_color: int) -> np.ndarray:
    area = scale * scale
    tie_color = random.choice([color for color in ALL_COLORS if color != major_color])
    block = np.full((scale, scale), int(major_color), dtype=int)
    for idx in random.sample(range(area), area // 2):
        r, c = divmod(idx, scale)
        block[r, c] = int(tie_color)
    status, _color, pair = _block_status(block)
    assert status == "tie" and pair == tuple(sorted((int(major_color), int(tie_color))))
    return block


def _make_breaker_block(scale: int, major_color: int) -> np.ndarray:
    alt = _noise_colors(major_color, 5)
    block = np.full((scale, scale), int(major_color), dtype=int)

    if scale == 4:
        block[0:2, 0:2] = np.array([[major_color, alt[0]], [alt[1], major_color]], dtype=int)
        block[0:2, 2:4] = np.array([[major_color, alt[2]], [alt[3], major_color]], dtype=int)
    elif scale == 6:
        left = np.array(
            [
                [major_color, alt[0], alt[1]],
                [alt[0], alt[1], major_color],
                [alt[1], major_color, alt[0]],
            ],
            dtype=int,
        )
        right = np.array(
            [
                [major_color, alt[2], alt[3]],
                [alt[2], alt[3], major_color],
                [alt[3], major_color, alt[2]],
            ],
            dtype=int,
        )
        block[0:3, 0:3] = left
        block[0:3, 3:6] = right
    else:
        raise AssertionError(f"unsupported scale {scale}")

    status, color, _ = _block_status(block)
    assert status == "majority" and color == major_color
    for divisor in _proper_divisors(scale):
        assert _local_invalidates(block, divisor)
    return block


def _local_invalidates(block: np.ndarray, scale: int) -> bool:
    height, width = map(int, block.shape)
    assert height % scale == 0 and width % scale == 0
    ties = 0
    for rr in range(0, height, scale):
        for cc in range(0, width, scale):
            status, _color, _pair = _block_status(block[rr : rr + scale, cc : cc + scale])
            if status == "invalid":
                return True
            if status == "tie":
                ties += 1
                if ties >= 2:
                    return True
    return False


def _dims_for_scale(scale: int) -> list[tuple[int, int]]:
    limit = 30 // scale
    dims = []
    for height in range(3, limit + 1):
        for width in range(4, limit + 1):
            if math.gcd(height, width) != 1:
                continue
            if height * width < 12:
                continue
            dims.append((height, width))
    assert dims
    return dims


def _random_symmetric_small(scale: int) -> np.ndarray:
    for _ in range(200):
        height, width = random.choice(_dims_for_scale(scale))
        palette = random.sample(OUTPUT_COLORS, random.randint(3, 5))
        small = np.zeros((height, width), dtype=int)
        for rr in range(height):
            for cc in range(width // 2):
                color = random.choice(palette)
                small[rr, cc] = color
                small[rr, width - 1 - cc] = color
            if width % 2 == 1:
                small[rr, width // 2] = random.choice(palette)
        if np.unique(small).size >= 2 and len({tuple(row) for row in small.tolist()}) >= 2:
            return small
    raise RuntimeError("failed to build a varied symmetric small grid")


def _pick_tie_position(shape: tuple[int, int]) -> tuple[int, int]:
    height, width = map(int, shape)
    return random.randrange(height), random.randrange(width // 2)


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    scale = random.choices(SCALES, weights=[0.45, 0.55], k=1)[0]
    clean = _random_symmetric_small(scale)
    height, width = map(int, clean.shape)
    tie_enabled = random.random() < TIE_PROBABILITY
    tie_pos = _pick_tie_position(clean.shape) if tie_enabled else None
    mirror_pos = None if tie_pos is None else (tie_pos[0], width - 1 - tie_pos[1])

    blocked = set()
    if tie_pos is not None:
        blocked.add(tie_pos)
        blocked.add(mirror_pos)
    breaker_candidates = [(rr, cc) for rr in range(height) for cc in range(width) if (rr, cc) not in blocked]
    breaker_pos = random.choice(breaker_candidates)

    large = np.zeros((height * scale, width * scale), dtype=int)
    for rr in range(height):
        for cc in range(width):
            color = int(clean[rr, cc])
            pos = (rr, cc)
            if pos == breaker_pos:
                block = _make_breaker_block(scale, color)
            elif tie_pos is not None and pos == tie_pos:
                block = _make_tie_block(scale, color)
            elif mirror_pos is not None and pos == mirror_pos:
                block = _make_majority_block(scale, color, max_noise=max(1, scale // 2))
            else:
                block = _make_majority_block(scale, color)
            large[rr * scale : (rr + 1) * scale, cc * scale : (cc + 1) * scale] = block

    meta = {
        "scale": scale,
        "tie_pos": tie_pos,
        "mirror_pos": mirror_pos,
        "breaker_pos": breaker_pos,
    }
    _validate_case(large, clean, meta)
    return large, clean, meta


def _validate_case(grid: np.ndarray, clean: np.ndarray, meta: dict) -> None:
    scale = int(meta["scale"])
    tie_pos = meta["tie_pos"]
    mirror_pos = meta["mirror_pos"]
    breaker_pos = meta["breaker_pos"]

    assert grid.ndim == 2 and clean.ndim == 2
    assert max(grid.shape) <= 30 and max(clean.shape) <= 30
    assert np.array_equal(clean, np.fliplr(clean))
    assert np.unique(clean).size >= 2

    pure = _scale_up(clean, scale)
    assert np.any(grid != pure)

    for rr in range(clean.shape[0]):
        for cc in range(clean.shape[1]):
            block = grid[rr * scale : (rr + 1) * scale, cc * scale : (cc + 1) * scale]
            assert np.any(block != int(clean[rr, cc]))

    candidates = _candidate_scales(grid.shape)
    analyses = {candidate: _analyze_scale(grid, candidate) for candidate in candidates}
    valid = [candidate for candidate in candidates if analyses[candidate]["valid"]]
    assert valid == [scale]
    assert np.array_equal(main(grid), clean)

    breaker_r, breaker_c = breaker_pos
    breaker_block = grid[breaker_r * scale : (breaker_r + 1) * scale, breaker_c * scale : (breaker_c + 1) * scale]
    for divisor in _proper_divisors(scale):
        assert _local_invalidates(breaker_block, divisor)

    if tie_pos is None:
        assert analyses[scale]["ties"] == []
    else:
        assert analyses[scale]["ties"] == [tie_pos]
        tie_r, tie_c = tie_pos
        tie_block = grid[tie_r * scale : (tie_r + 1) * scale, tie_c * scale : (tie_c + 1) * scale]
        status, _color, pair = _block_status(tie_block)
        assert status == "tie"
        assert int(clean[tie_pos]) in pair

        mirror_r, mirror_c = mirror_pos
        mirror_block = grid[mirror_r * scale : (mirror_r + 1) * scale, mirror_c * scale : (mirror_c + 1) * scale]
        mirror_status, mirror_color, _pair = _block_status(mirror_block)
        assert mirror_status == "majority"
        assert mirror_color == int(clean[tie_pos]) == int(clean[mirror_pos])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the unique noisy upscale factor and compress the image back to its clean mirror-symmetric small form.

    Input structure:
    - The whole grid is a uniformly upscaled version of a smaller image, using one unknown integer block scale.
    - Each small-image cell became a same-color square block, but each block contains a small amount of color noise.
    - Exactly one common divisor greater than 1 makes every block have a strict majority color, except that at most one block may be an exact two-color tie.

    Rule:
    - Enumerate every common divisor greater than 1 of the grid height and width, and keep the unique scale whose block partition satisfies the majority condition.
    - Compress the grid by replacing each block with its majority color.
    - If one block is tied, resolve it with the restored small image's left-right mirror symmetry by copying the color from the tied block's horizontal mirror partner.
    - Output the restored small image.

    Key insight:
    - The scale is not marked geometrically; it must be inferred from block-level color statistics, and the only remaining ambiguity is broken by the final small image being horizontally mirror symmetric.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    valid = []
    for scale in _candidate_scales(g.shape):
        analysis = _analyze_scale(g, scale)
        if analysis["valid"]:
            valid.append(analysis)

    assert len(valid) == 1, "expected a unique valid upscale factor"
    return valid[0]["colors"]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, _clean, _meta = _generate_case()
            return grid
        except (AssertionError, RuntimeError):
            continue
    raise RuntimeError("failed to generate a valid input")
